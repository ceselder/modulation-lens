"""Stage 2: NNOMP-decompose 500k real L42 activations against the <=12-token bank, emit top-1
bullet targets for AV SFT.

NNOMP, not plain MP: every greedy step re-solves the NON-NEGATIVE least squares over the WHOLE
selected support, so earlier weights are corrected as later atoms are added. That is what makes
"top bullet" ambiguous and worth measuring -- the first-PICKED atom (max correlation with the
un-reduced target) is frequently not the highest-WEIGHTED one after refit. We emit the
highest-weighted atom and report the disagreement rate.

Space: whitened J-space, matching inv_train's reward exactly.
  phrase side : (v_raw @ J.T) @ W.T      v_raw already carries the bank's cell-mean subtraction
  target side : ((a @ J.T) - mu) @ W.T   mu is the activation-pool mean from the whitener
Two DIFFERENT means, which is the configuration this project measured as load-bearing: one shared
mean leaves a blank string at 0.259 cosine, two means put it at 0.008.
"""
import os
import modal

app = modal.App("celeste-stage2-nnomp")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("torch==2.8.0", "numpy", "pyarrow")
       .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200", cpu=8.0, memory=196608, timeout=14400)
def decompose(k: int = 4, max_tokens: int = 12, batch: int = 256, limit: int = 0,
              whiten_ridge: str = "0.01", whiten: int = 1, center: int = 1,
              affine: str = "", out: str = "/vol/data/nnomp_top1_12tok.jsonl"):
    import glob, itertools, json, time
    import numpy as np, torch, pyarrow.parquet as pq

    dev = "cuda"
    jp = glob.glob("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/*/"
                   "qwen3.6-27b/j-lens/lens.pt")
    if not jp:
        raise SystemExit("j-lens not in the cache")
    J = torch.load(jp[0], map_location="cpu", weights_only=False)["J"][42].to(dev).float()
    z = np.load("/vol/data/natural_whitener_jspace.npz")
    if int(z.get("jtransformed", 0)) != 1:
        raise SystemExit("whitener is not J-transformed")
    keys = [kk for kk in z.files if kk.startswith("W_ridge")]
    wk = "W_ridge%s" % whiten_ridge
    if wk not in z.files:
        wk = sorted(keys)[0]
        print("[whiten] ridge %s absent, using %s (available: %s)"
              % (whiten_ridge, wk, keys), flush=True)
    mu = torch.tensor(z["mu"], device=dev).float()
    W = torch.tensor(z[wk], device=dev).float()
    print("[space] whiten=%d center=%d | %s | mu norm %.3f"
          % (whiten, center, wk, float(mu.norm())), flush=True)

    # The fitted alignment between the modulation-read distribution and the natural-activation
    # distribution. Without it the same decomposition scores 0.360 instead of 0.633 (1.76x).
    M = None
    if affine:
        M = torch.from_numpy(np.load(affine)).to(dev).float()
        print("[affine] loaded %s %s" % (affine, tuple(M.shape)), flush=True)

    labels = json.load(open("/vol/pg_dict/labels.json"))
    Vf = np.load("/vol/pg_dict/vectors.npy", mmap_mode="r")
    nt = np.load("/vol/pg_dict/n_tokens.npy")
    sel = np.nonzero(nt <= max_tokens)[0]
    labels = [labels[i] for i in sel]
    N = len(sel)
    print("[bank] %s atoms at <=%d tokens (of %s)" % (f"{N:,}", max_tokens, f"{len(nt):,}"),
          flush=True)

    # atoms -> whitened J-space, unit norm; filled in slices so no full fp32 copy ever exists
    A = torch.empty((N, 5120), dtype=torch.float16, device=dev)
    CH = 200_000
    for a in range(0, N, CH):
        blk = torch.from_numpy(np.ascontiguousarray(Vf[sel[a:a + CH]])).to(dev, torch.float32)
        bw = (blk @ J.T) @ W.T if whiten else (blk @ J.T)
        if M is not None: bw = bw @ M.T
        A[a:a + CH] = (bw / bw.norm(dim=1, keepdim=True).clamp(min=1e-8)).half()
        del blk, bw
    torch.cuda.empty_cache()
    print("[bank] projected to whitened J-space (%.1f GB)" % (A.numel() * 2 / 1e9), flush=True)

    pf = pq.ParquetFile("/vol/data/prose_L42_500k.parquet")
    nrows = pf.metadata.num_rows if not limit else min(limit, pf.metadata.num_rows)
    print("[data] %s activations" % f"{nrows:,}", flush=True)

    # exact NNLS over a <=k support: enumerate every non-empty subset, solve each batched, keep the
    # best feasible one. Clamping an unconstrained lstsq would NOT be exact.
    supports = [s for r in range(1, k + 1) for s in itertools.combinations(range(k), r)]

    def nnls_batch(B, t):
        """B [b,k,d] chosen atoms, t [b,d] unit targets -> w [b,k] exact non-negative weights."""
        b = B.shape[0]
        G = B @ B.transpose(1, 2)                       # [b,k,k]
        c = torch.einsum("bkd,bd->bk", B, t)            # [b,k]
        bestw = torch.zeros((b, k), device=dev)
        bestr = torch.full((b,), -1e9, device=dev)
        for sup in supports:
            idx = torch.tensor(sup, device=dev)
            Gs = G[:, idx][:, :, idx]
            cs = c[:, idx]
            Gs = Gs + 1e-6 * torch.eye(len(sup), device=dev).unsqueeze(0)
            try:
                ws = torch.linalg.solve(Gs, cs.unsqueeze(-1)).squeeze(-1)
            except Exception:
                continue
            feas = (ws >= -1e-8).all(dim=1)
            ws = ws.clamp(min=0.0)
            # objective: cosine of the reconstruction with the target
            rec = torch.einsum("bk,bkd->bd", ws, B[:, idx])
            rn = rec.norm(dim=1).clamp(min=1e-8)
            val = torch.einsum("bd,bd->b", rec, t) / rn
            val = torch.where(feas, val, torch.full_like(val, -1e9))
            upd = val > bestr
            bestr = torch.where(upd, val, bestr)
            full = torch.zeros((b, k), device=dev)
            full[:, idx] = ws
            bestw = torch.where(upd.unsqueeze(1), full, bestw)
        return bestw, bestr

    fh = open(out, "w")
    n_done = n_disagree = 0
    fve_sum = 0.0
    t0 = time.time()
    row0 = 0
    for bt in pf.iter_batches(batch_size=batch, columns=["activation_vector", "label",
                                                         "ctx", "doc_id", "pos"]):
        d = bt.to_pydict()
        b = len(d["label"])
        if limit and row0 >= limit: break
        acts = torch.from_numpy(np.asarray(
            bt.column("activation_vector").flatten().to_numpy(zero_copy_only=False),
            dtype="float32").reshape(b, 5120)).to(dev)
        tj = (acts @ J.T)
        if center: tj = tj - mu
        t = tj @ W.T if whiten else tj
        t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)

        resid = t.clone()
        chosen = torch.zeros((b, k), dtype=torch.long, device=dev)
        for step in range(k):
            sims = (A @ resid.half().T).float()          # [N, b] -- no gather of A
            for pj in range(step):
                sims[chosen[:, pj], torch.arange(b, device=dev)] = -1e9
            pick = sims.argmax(dim=0)
            chosen[:, step] = pick
            Bsel = A[chosen[:, :step + 1].reshape(-1)].float().reshape(b, step + 1, 5120)
            if step + 1 < k:
                wpart = torch.linalg.lstsq(Bsel.transpose(1, 2), t.unsqueeze(-1)).solution
                wpart = wpart.squeeze(-1).clamp(min=0.0)
                resid = t - torch.einsum("bk,bkd->bd", wpart, Bsel)
            del sims
        Bsel = A[chosen.reshape(-1)].float().reshape(b, k, 5120)
        w, rval = nnls_batch(Bsel, t)

        first = chosen[:, 0]
        topw = chosen.gather(1, w.argmax(dim=1, keepdim=True)).squeeze(1)
        n_disagree += int((first != topw).sum())
        fve_sum += float((rval.clamp(min=0) ** 2).sum())

        fw, tw, ww, rr = (first.tolist(), topw.tolist(), w.tolist(), rval.tolist())
        ch = chosen.tolist()
        for j in range(b):
            fh.write(json.dumps({
                "i": row0 + j, "label": d["label"][j], "doc_id": d["doc_id"][j],
                "pos": d["pos"][j],
                "bullets": [labels[tw[j]]],                    # the SFT target: top by WEIGHT
                "top_by_weight": labels[tw[j]],
                "top_by_first_pick": labels[fw[j]],
                "all_atoms": [labels[c] for c in ch[j]],
                "weights": [round(x, 5) for x in ww[j]],
                "fve": round(max(rr[j], 0.0) ** 2, 5),
            }) + "\n")
        n_done += b; row0 += b
        if (row0 // batch) % 40 == 0:
            print("[nnomp] %s/%s | mean FVE %.4f | first!=topweight %.1f%% | %.0f act/s"
                  % (f"{n_done:,}", f"{nrows:,}", fve_sum / max(n_done, 1),
                     100 * n_disagree / max(n_done, 1), n_done / max(time.time() - t0, 1e-9)),
                  flush=True)
    fh.close()
    rep = {"n": n_done, "k": k, "max_tokens": max_tokens, "bank_atoms": int(N),
           "mean_fve": fve_sum / max(n_done, 1),
           "frac_first_ne_topweight": n_disagree / max(n_done, 1), "out": out,
           "whiten": int(whiten), "center": int(center), "affine": affine}
    json.dump(rep, open("/vol/data/nnomp_top1_report.json", "w"), indent=1)
    VOL.commit()
    print("[final] %s rows | mean FVE %.4f | first-pick != top-weight on %.1f%%"
          % (f"{n_done:,}", rep["mean_fve"], 100 * rep["frac_first_ne_topweight"]), flush=True)
    print("NNOMP_DONE", flush=True)
    return rep
