"""Stage 1a: build the AR training set (atom text -> modulation vector) and fit the ridge reference.

The AR maps a phrase to the L42 modulation vector it produces. The reference to beat is a ridge
from bag-of-token-embeddings, which this project already measured at 0.685 cos (75% of the 0.911
two-measurement ceiling) -- because v(phrase) is close to a weighted sum of its token embeddings.
Anything the LoRA AR adds has to show up ON TOP of that, so the ridge number is computed here on
exactly the same split rather than quoted from memory.

Atoms over MAXTOK tokens are dropped: the 13-16 token tail is not coherent enough to be an atom.
"""
import os
import modal

app = modal.App("celeste-ar-ridge")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "numpy", "pyarrow", "safetensors",
                    "accelerate", "huggingface_hub")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
MODEL = "Qwen/Qwen3.6-27B"


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200", cpu=8.0, memory=131072, timeout=7200)
def build(max_tokens: int = 12, n_eval: int = 20000, ridge_lambdas: str = "1,10,100,1000"):
    import json, numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    labels = json.load(open("/vol/pg_dict/labels.json"))
    V = np.load("/vol/pg_dict/vectors.npy", mmap_mode="r")
    nt = np.load("/vol/pg_dict/n_tokens.npy")
    rho = np.load("/vol/pg_dict/rho.npy")
    print("[in] %s atoms, vectors %s" % (f"{len(labels):,}", V.shape), flush=True)

    keep = np.nonzero(nt <= max_tokens)[0]
    print("[len] <=%d tokens keeps %s of %s (%.1f%%); dropped tail mean nt %.2f"
          % (max_tokens, f"{len(keep):,}", f"{len(nt):,}", 100 * len(keep) / len(nt),
             float(nt[nt > max_tokens].mean()) if (nt > max_tokens).any() else 0.0), flush=True)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(keep))
    keep = keep[perm]                      # shuffle so the split is not domain-ordered
    ev, tr = keep[:n_eval], keep[n_eval:]
    print("[split] train %s | eval %s" % (f"{len(tr):,}", f"{len(ev):,}"), flush=True)

    # --- ceiling on this split -----------------------------------------------------------------
    # rho is the mean pairwise cosine BETWEEN grid cells for one span. sqrt(rho) would be the bound
    # for predicting a SINGLE cell draw -- but the stored target is the MEAN of all NCELL cells
    # (VEC = (X - MU).mean(axis=1) in reliability_pass), so its noise variance is NCELL times
    # smaller and the achievable ceiling is far higher. Using this project's own conversion,
    # S = rho/(1-rho) per cell and k-draw accuracy sqrt(kS/(kS+1)):
    NCELL = 16
    _r = np.clip(rho[ev].astype("float64"), 1e-6, 1 - 1e-6)
    _S = _r / (1 - _r)
    bound_1 = float(np.sqrt(_S / (_S + 1)).mean())
    bound_k = float(np.sqrt(NCELL * _S / (NCELL * _S + 1)).mean())
    print("[ceiling] eval rho mean %.4f | k=1 bound %.4f (NOT the right one) | k=%d bound %.4f"
          % (float(_r.mean()), bound_1, NCELL, bound_k), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    # MUST be AutoModelForCausalLM. AutoModel instantiates a skeleton whose parameter names lack
    # the checkpoint's "model." prefix, so EVERY weight is rejected as UNEXPECTED and the model is
    # randomly initialised -- and a ridge on random 5120-d projections of a token bag still returns
    # a believable ~0.5 cosine, so the failure is invisible in the metric. Guard it explicitly.
    full, info = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    miss = [k for k in info.get("missing_keys", []) if "lora" not in k]
    if miss:
        raise SystemExit("REFUSING to run: %d weights did not load, e.g. %s"
                         % (len(miss), miss[:4]))
    print("[load] all weights loaded (0 missing, %d unexpected)"
          % len(info.get("unexpected_keys", [])), flush=True)
    emb = full.get_input_embeddings()
    E = emb.weight.detach().float()                       # [vocab, 5120]
    print("[emb] input embedding matrix %s" % (tuple(E.shape),), flush=True)
    D = E.shape[1]

    def bag(idx, chunk=20000):
        """mean of input embeddings over the atom's tokens -- the ridge feature."""
        out = torch.zeros((len(idx), D), dtype=torch.float32, device="cuda")
        for a in range(0, len(idx), chunk):
            batch = [labels[i] for i in idx[a:a + chunk]]
            ids = tok(batch, add_special_tokens=False)["input_ids"]
            for j, seq in enumerate(ids):
                if not seq: continue
                t = torch.tensor(seq, device="cuda")
                out[a + j] = E[t].mean(0)
        return out

    Ytr = torch.from_numpy(np.ascontiguousarray(V[np.sort(tr)])).to("cuda", torch.float32)
    Yev = torch.from_numpy(np.ascontiguousarray(V[np.sort(ev)])).to("cuda", torch.float32)
    Xtr, Xev = bag(np.sort(tr)), bag(np.sort(ev))
    print("[feat] Xtr %s Ytr %s" % (tuple(Xtr.shape), tuple(Ytr.shape)), flush=True)

    XtX = Xtr.T @ Xtr
    XtY = Xtr.T @ Ytr
    I = torch.eye(D, device="cuda")
    Yev_n = Yev / Yev.norm(dim=1, keepdim=True).clamp(min=1e-8)
    best = None
    for lam in [float(x) for x in ridge_lambdas.split(",")]:
        W = torch.linalg.solve(XtX + lam * I, XtY)
        P = Xev @ W
        cos = float(((P / P.norm(dim=1, keepdim=True).clamp(min=1e-8)) * Yev_n).sum(1).mean())
        print("[ridge] lambda %-7g held-out cos %.4f" % (lam, cos), flush=True)
        if best is None or cos > best[0]: best = (cos, lam, W)
    print("[ridge] BEST held-out cos %.4f at lambda %g" % (best[0], best[1]), flush=True)

    os.makedirs("/vol/ar_stage1", exist_ok=True)
    np.save("/vol/ar_stage1/ridge_W.npy", best[2].cpu().numpy().astype("float32"))
    np.save("/vol/ar_stage1/split_eval_idx.npy", np.sort(ev))
    np.save("/vol/ar_stage1/split_train_idx.npy", np.sort(tr))
    json.dump({"max_tokens": max_tokens, "n_train": int(len(tr)), "n_eval": int(len(ev)),
               "ridge_cos": best[0], "ridge_lambda": best[1],
               "eval_rho_mean": float(_r.mean()),
               "cos_bound": bound_k, "cos_bound_k": NCELL,
               "cos_bound_single_draw": bound_1},
              open("/vol/ar_stage1/ridge_report.json", "w"), indent=1)
    VOL.commit()
    print("RIDGE_DONE %.4f" % best[0], flush=True)
    return best[0]
