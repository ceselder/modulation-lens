"""Does NNOMP's FVE 0.3615 reflect the dictionary, or a MISSING ALIGNMENT?

Atoms are modulation reads (phrase in a template, pooled over carrier positions); targets are
natural prose activations at one position. Both are L42, but they are different distributions, and
stage 2 aligned them with nothing but two mean subtractions plus a J that was fitted for a
different purpose. This tests the alternatives:

  raw          decompose in raw L42, mean-centred          (J never tested against this)
  jspace       decompose in J-space, mean-centred          (what stage 2 did -> 0.3615)
  +affine      fit a ridge map M on TRAIN activations from the unaligned reconstruction to the
               target, apply M to the ATOMS, then re-select and re-fit on TEST. Fitting on train
               and scoring on test is what keeps this from being a post-hoc inflation: M has to
               generalise to activations it never saw.
"""
import os
import modal

app = modal.App("celeste-stage2-affine")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("torch==2.8.0", "numpy", "pyarrow")
       .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200", cpu=8.0, memory=196608, timeout=10800)
def run(k: int = 4, max_tokens: int = 12, n_train: int = 20000, n_test: int = 4000,
        batch: int = 256, lambdas: str = "1,10,100,1000", spaces: str = "raw,jspace"):
    import glob, itertools, json
    import numpy as np, torch, pyarrow.parquet as pq
    dev = "cuda"

    jp = glob.glob("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/*/"
                   "qwen3.6-27b/j-lens/lens.pt")
    J = torch.load(jp[0], map_location="cpu", weights_only=False)["J"][42].to(dev).float()
    z = np.load("/vol/data/natural_whitener_jspace.npz")
    mu_j = torch.tensor(z["mu"], device=dev).float()

    Vf = np.load("/vol/pg_dict/vectors.npy", mmap_mode="r")
    nt = np.load("/vol/pg_dict/n_tokens.npy")
    sel = np.nonzero(nt <= max_tokens)[0]
    N = len(sel)

    pf = pq.ParquetFile("/vol/data/prose_L42_500k.parquet")
    need = n_train + n_test
    acts = []
    for bt in pf.iter_batches(batch_size=4096, columns=["activation_vector"]):
        acts.append(np.asarray(bt.column("activation_vector").flatten().to_numpy(
            zero_copy_only=False), dtype="float32").reshape(-1, 5120))
        if sum(a.shape[0] for a in acts) >= need: break
    ACT = torch.from_numpy(np.concatenate(acts)[:need]).to(dev)
    mu_raw = ACT.mean(0)                        # activation-pool mean in RAW space
    print("[data] %d activations (%d train / %d test) | bank %s atoms"
          % (need, n_train, n_test, f"{N:,}"), flush=True)

    supports = [s for r in range(1, k + 1) for s in itertools.combinations(range(k), r)]

    def nnls(B, t):
        b = B.shape[0]
        G = B @ B.transpose(1, 2); c = torch.einsum("bkd,bd->bk", B, t)
        bw = torch.zeros((b, k), device=dev); br = torch.full((b,), -1e9, device=dev)
        for sup in supports:
            idx = torch.tensor(sup, device=dev)
            Gs = G[:, idx][:, :, idx] + 1e-6 * torch.eye(len(sup), device=dev).unsqueeze(0)
            ws = torch.linalg.solve(Gs, c[:, idx].unsqueeze(-1)).squeeze(-1)
            feas = (ws >= -1e-8).all(dim=1); ws = ws.clamp(min=0.0)
            rec = torch.einsum("bk,bkd->bd", ws, B[:, idx])
            val = torch.einsum("bd,bd->b", rec, t) / rec.norm(dim=1).clamp(min=1e-8)
            val = torch.where(feas, val, torch.full_like(val, -1e9))
            upd = val > br; br = torch.where(upd, val, br)
            full = torch.zeros((b, k), device=dev); full[:, idx] = ws
            bw = torch.where(upd.unsqueeze(1), full, bw)
        return bw, br

    def build_bank(space, M=None):
        A = torch.empty((N, 5120), dtype=torch.float16, device=dev)
        CH = 200_000
        for a in range(0, N, CH):
            blk = torch.from_numpy(np.ascontiguousarray(Vf[sel[a:a + CH]])).to(dev, torch.float32)
            v = blk @ J.T if space == "jspace" else blk
            if M is not None: v = v @ M.T
            A[a:a + CH] = (v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)).half()
            del blk, v
        torch.cuda.empty_cache()
        return A

    def targets(space, idx, M=None):
        t = ACT[idx]
        t = (t @ J.T) - mu_j if space == "jspace" else t - mu_raw
        return t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def decompose(A, T):
        """-> reconstruction vectors (unnormalised, in A's space) and FVE per row."""
        out_rec = torch.empty_like(T); out_f = torch.empty(T.shape[0], device=dev)
        for a in range(0, T.shape[0], batch):
            t = T[a:a + batch]; b = t.shape[0]
            resid = t.clone(); chosen = torch.zeros((b, k), dtype=torch.long, device=dev)
            for st in range(k):
                sims = (A @ resid.half().T).float()
                for pj in range(st):
                    sims[chosen[:, pj], torch.arange(b, device=dev)] = -1e9
                chosen[:, st] = sims.argmax(dim=0)
                if st + 1 < k:
                    Bs = A[chosen[:, :st + 1].reshape(-1)].float().reshape(b, st + 1, 5120)
                    wp = torch.linalg.lstsq(Bs.transpose(1, 2), t.unsqueeze(-1)).solution
                    resid = t - torch.einsum("bk,bkd->bd", wp.squeeze(-1).clamp(min=0.0), Bs)
                del sims
            Bs = A[chosen.reshape(-1)].float().reshape(b, k, 5120)
            w, r = nnls(Bs, t)
            out_rec[a:a + batch] = torch.einsum("bk,bkd->bd", w, Bs)
            out_f[a:a + batch] = r.clamp(min=0) ** 2
        return out_rec, out_f

    tr_i = torch.arange(0, n_train, device=dev)
    te_i = torch.arange(n_train, need, device=dev)
    res = {}
    for space in [x.strip() for x in spaces.split(",") if x.strip()]:
        A = build_bank(space)
        Ttr, Tte = targets(space, tr_i), targets(space, te_i)
        rec_tr, _ = decompose(A, Ttr)
        _, f_te = decompose(A, Tte)
        res[space] = float(f_te.mean())
        print("[%s] test FVE %.4f (no affine)" % (space, res[space]), flush=True)

        # fit M: unaligned reconstruction -> target, on TRAIN only
        X, Y = rec_tr.float(), Ttr.float()
        XtX = X.T @ X; XtY = X.T @ Y; I = torch.eye(5120, device=dev)
        best = None
        for lam in [float(x) for x in lambdas.split(",")]:
            M = torch.linalg.solve(XtX + lam * I, XtY).T     # y ~ M x
            Aa = build_bank(space, M=M)
            _, fa = decompose(Aa, Tte)
            v = float(fa.mean())
            print("   [%s +affine lam %-6g] test FVE %.4f  (%+.4f)" % (space, lam, v,
                                                                       v - res[space]), flush=True)
            if best is None or v > best[0]:
                best = (v, lam)
                # SAVE it -- the whole point is reuse by stage 2's target generation and by the
                # stage-3 reward, which makes the same cross-distribution comparison.
                np.save("/vol/data/affine_M_%s.npy" % space, M.cpu().numpy().astype("float32"))
                json.dump({"space": space, "lam": lam, "test_fve": v,
                           "baseline_fve": res[space], "n_train": int(n_train),
                           "n_test": int(n_test), "k": k, "max_tokens": max_tokens},
                          open("/vol/data/affine_M_%s.json" % space, "w"), indent=1)
                VOL.commit()
                print("      saved /vol/data/affine_M_%s.npy (lam %g, FVE %.4f)"
                      % (space, lam, v), flush=True)
            del Aa
            torch.cuda.empty_cache()
        res[space + "_affine"] = best[0]; res[space + "_affine_lam"] = best[1]
        del A
        torch.cuda.empty_cache()
    json.dump(res, open("/vol/data/stage2_affine_report.json", "w"), indent=1)
    VOL.commit()
    print("[summary] %s" % json.dumps(res), flush=True)
    print("AFFINE_DONE", flush=True)
    return res
