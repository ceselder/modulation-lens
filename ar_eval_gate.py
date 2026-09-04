"""Two gates the AR must pass before it becomes a frozen RL reward.

GATE 1 -- generalisation, not memorisation. Our bank was exactly deduped and near-dup filtered in
3 tiers, but near-duplicate spans could still straddle the train/eval split, which would let a
0.859 held-out cosine be recall rather than generalisation. thinkies-v3 labels come from a
completely different construction, so they cannot overlap: if the AR still predicts those well, it
generalises.

GATE 2 -- RANK agreement with the true reward. This is the gate that actually matters. GRPO
normalises advantages inside a group, so a surrogate that is biased but correctly ORDERED trains
fine, while an unbiased one that misranks injects pure gradient noise. The eval atoms already carry
their MEASURED 16-cell vectors, so the true reward ordering is available with no grid reads: rank
atoms against a target by measured-vector cosine, rank them again by AR-predicted cosine, and
compare. Spearman plus top-1/top-4 agreement is exactly what the advantage sees.
"""
import os
import modal

app = modal.App("celeste-ar-gate")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "numpy", "pyarrow", "peft",
                    "safetensors", "accelerate", "scipy")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
MODEL = "Qwen/Qwen3.6-27B"
READ_LAYER = 42


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200", cpu=8.0, memory=131072, timeout=7200)
def gate(ar_dir: str = "/vol/ar_l42_text2vec", n_atoms: int = 20000, n_targets: int = 256,
         n_thinkies: int = 20000, max_tokens: int = 12):
    import glob, json
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from scipy.stats import spearmanr
    import pyarrow.parquet as pq

    meta = json.load(open(ar_dir + "/ar_meta.json"))
    print("[ar] checkpoint step %d, held-out cos %.4f (ridge %.4f, bound %.4f)"
          % (meta["step"], meta["heldout_cos"], meta["ridge_cos"], meta["cos_bound"]), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    _full, info = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    miss = [k for k in info.get("missing_keys", []) if "lora" not in k]
    if miss: raise SystemExit("weights did not load: %s" % miss[:3])
    base = _full.model
    base.layers = base.layers[:READ_LAYER + 1]
    del _full.lm_head
    model = PeftModel.from_pretrained(base, ar_dir).eval()
    hd = torch.load(ar_dir + "/head.pt", map_location="cuda")
    head = torch.nn.Linear(5120, 5120, bias=True).to("cuda", torch.float32)
    head.load_state_dict(hd["head"]); head.eval()
    HK = {"h": None}
    # PeftModel wraps the module; reach the real layer list for the hook
    lay = model.base_model.model.layers if hasattr(model, "base_model") else model.layers
    lay[READ_LAYER].register_forward_hook(
        lambda mm, i, o: HK.__setitem__("h", o[0] if isinstance(o, tuple) else o))

    @torch.no_grad()
    def predict(texts, bs=128):
        out = torch.empty((len(texts), 5120), dtype=torch.float32, device="cuda")
        for a in range(0, len(texts), bs):
            b = tok(texts[a:a + bs], add_special_tokens=False, padding=True, truncation=True,
                    max_length=max_tokens + 2, return_tensors="pt").to("cuda")
            model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
            h = HK["h"]
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            out[a:a + bs] = head(((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).float())
        return out

    def cosmean(P, Y):
        Pn = P / P.norm(dim=1, keepdim=True).clamp(min=1e-8)
        Yn = Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return float((Pn * Yn).sum(1).mean())

    # ---------------- GATE 1: in-distribution eval atoms, then OOD thinkies -------------------
    labels = json.load(open("/vol/pg_dict/labels.json"))
    V = np.load("/vol/pg_dict/vectors.npy", mmap_mode="r")
    ev = np.load("/vol/ar_stage1/split_eval_idx.npy")[:n_atoms]
    Yev = torch.from_numpy(np.ascontiguousarray(V[ev])).to("cuda", torch.float32)
    Pev = predict([labels[i] for i in ev])
    print("[gate1] in-distribution (%d held-out atoms): cos %.4f" % (len(ev), cosmean(Pev, Yev)),
          flush=True)

    tl, tv = [], []
    for sh in sorted(glob.glob("/vol/thinkies_v3/thinkies_v3-*-of-00007.parquet")):
        for b in pq.ParquetFile(sh).iter_batches(batch_size=8192, columns=["label", "vector"]):
            tl += b.column("label").to_pylist()
            tv.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                                 dtype="float32").reshape(-1, 5120))
            if len(tl) >= n_thinkies: break
        if len(tl) >= n_thinkies: break
    tl, Yt = tl[:n_thinkies], torch.from_numpy(np.concatenate(tv)[:n_thinkies]).to("cuda")
    Pt = predict(tl)
    print("[gate1] OOD thinkies-v3 (%d labels, different construction): cos %.4f"
          % (len(tl), cosmean(Pt, Yt)), flush=True)
    print("[gate1] NOTE thinkies vectors carry a different centring constant, so this is a FLOOR "
          "on OOD quality, not a like-for-like number.", flush=True)

    # ---------------- GATE 2: rank agreement with the true reward ------------------------------
    pf = pq.ParquetFile("/vol/data/prose_L42_500k.parquet")
    bt = next(pf.iter_batches(batch_size=n_targets, columns=["activation_vector"]))
    acts = torch.from_numpy(np.asarray(
        bt.column("activation_vector").flatten().to_numpy(zero_copy_only=False),
        dtype="float32").reshape(-1, 5120)).to("cuda")
    jp = glob.glob("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/*/"
                   "qwen3.6-27b/j-lens/lens.pt")
    J = torch.load(jp[0], map_location="cpu", weights_only=False)["J"][42].to("cuda").float()
    z = np.load("/vol/data/natural_whitener_jspace.npz")
    mu = torch.tensor(z["mu"], device="cuda").float()
    # the space chosen by measurement: unwhitened J, mean-centred (whitening cost 6.3x FVE)
    T = (acts @ J.T) - mu
    T = T / T.norm(dim=1, keepdim=True).clamp(min=1e-8)
    def toJ(X):
        Xj = X @ J.T
        return Xj / Xj.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Atrue, Apred = toJ(Yev), toJ(Pev)

    Strue = (Atrue @ T.T).T.cpu().numpy()      # [targets, atoms] true reward
    Spred = (Apred @ T.T).T.cpu().numpy()      # [targets, atoms] AR-surrogate reward
    rho_s, t1, t4, g16 = [], [], [], []
    rng = np.random.default_rng(0)
    for i in range(Strue.shape[0]):
        rho_s.append(spearmanr(Strue[i], Spred[i]).statistic)
        t1.append(int(Strue[i].argmax() == Spred[i].argmax()))
        a4 = set(np.argsort(-Strue[i])[:4]); b4 = set(np.argsort(-Spred[i])[:4])
        t4.append(len(a4 & b4) / 4.0)
        # GRPO sees groups of 16: does the surrogate pick the same winner inside a random group?
        gi = rng.choice(Strue.shape[1], size=16, replace=False)
        g16.append(int(Strue[i][gi].argmax() == Spred[i][gi].argmax()))
    rep = {"ar_step": meta["step"], "ar_heldout": meta["heldout_cos"],
           "cos_in_dist": cosmean(Pev, Yev), "cos_ood_thinkies": cosmean(Pt, Yt),
           "spearman_mean": float(np.mean(rho_s)), "top1_agree": float(np.mean(t1)),
           "top4_overlap": float(np.mean(t4)), "group16_argmax_agree": float(np.mean(g16)),
           "n_atoms": int(len(ev)), "n_targets": int(Strue.shape[0])}
    print("[gate2] over %d targets x %d atoms:" % (rep["n_targets"], rep["n_atoms"]), flush=True)
    print("        Spearman(true, AR)        %.4f" % rep["spearman_mean"], flush=True)
    print("        top-1 argmax agreement    %.3f" % rep["top1_agree"], flush=True)
    print("        top-4 set overlap         %.3f" % rep["top4_overlap"], flush=True)
    print("        group-of-16 winner match  %.3f   <-- what GRPO actually sees"
          % rep["group16_argmax_agree"], flush=True)
    json.dump(rep, open("/vol/ar_stage1/gate_report.json", "w"), indent=1)
    VOL.commit()
    print("GATE_DONE", flush=True)
    return rep
