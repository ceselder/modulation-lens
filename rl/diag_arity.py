"""Where does the AR's fidelity go when it enters the reward's comparison space?

The AR scores held-out cos ~0.91 against MEASURED atom vectors in RAW L42. The reward compares in
J-then-affine space. J is ill-conditioned (cond 70,646; participation ratio 2260/5120), so a
raw-space cosine need not survive it: the map amplifies whatever directions the AR's residual
error happens to occupy. This attributes the loss stage by stage on the SAME atoms:

    cos_raw   cos(v_pred,            v_meas)              expect ~0.91
    cos_J     cos(v_pred @ J.T,      v_meas @ J.T)        J alone
    cos_JM    cos(v_pred @ J.T @ M.T, v_meas @ J.T @ M.T) J then the affine  <- what the reward uses

If cos_JM is ~0.4, the AR is being trained on the wrong objective and its loss should be computed
in the comparison space instead. Also reports the same for a RIDGE prediction, to show whether
this is an AR property or a property of the map.
"""
import modal
app = modal.App("celeste-diag-arity")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "numpy", "pyarrow", "peft", "safetensors",
                    "accelerate")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200", cpu=8.0, memory=131072, timeout=5400)
def run(n: int = 512, ar_dir: str = "/vol/ar_l42_text2vec", max_tokens: int = 12):
    """Does the reward's attenuation grow with the number of bullets?

    Each atom is individually 0.93 cos accurate in the comparison space, yet the 4-atom composition
    falls from 0.804 (measured vectors) to 0.329 (AR-predicted). Hypothesis: NNLS composition
    AMPLIFIES per-atom error, because greedy selects each atom to cover the previous residual and
    that complementarity is destroyed by a 21-degree perturbation. If so the measured-vs-predicted
    gap should GROW with k, and k=1 should retain nearly all of the AR's fidelity.
    """
    import glob, itertools, json
    import numpy as np, torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import pyarrow.parquet as pq

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    m, info = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    assert not [q for q in info.get("missing_keys", []) if "lora" not in q]
    base = m.model; base.layers = base.layers[:43]; del m.lm_head
    model = PeftModel.from_pretrained(base, ar_dir).eval()
    hd = torch.load(ar_dir + "/head.pt", map_location="cuda")
    head = torch.nn.Linear(5120, 5120, bias=True).to("cuda", torch.float32)
    head.load_state_dict(hd["head"]); head.eval()
    HK = {}
    model.base_model.model.layers[42].register_forward_hook(
        lambda mm, i, o: HK.__setitem__("h", o[0] if isinstance(o, tuple) else o))

    @torch.no_grad()
    def pred(texts, bs=128):
        out = torch.empty((len(texts), 5120), device="cuda")
        for a in range(0, len(texts), bs):
            b = tok(texts[a:a+bs], add_special_tokens=False, padding=True, truncation=True,
                    max_length=max_tokens + 2, return_tensors="pt").to("cuda")
            model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
            h = HK["h"]; msk = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            out[a:a+bs] = head(((h * msk).sum(1) / msk.sum(1).clamp(min=1e-6)).float())
        return out

    jp = glob.glob("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/*/"
                   "qwen3.6-27b/j-lens/lens.pt")[0]
    J = torch.load(jp, map_location="cpu", weights_only=False)["J"][42].to("cuda").float()
    M = torch.from_numpy(np.load("/vol/data/affine_M_jspace.npy")).to("cuda").float()
    z = np.load("/vol/data/natural_whitener_jspace.npz")
    mu = torch.tensor(z["mu"], device="cuda").float()

    rows = []
    with open("/vol/data/nnomp_top12_12tok_affine.jsonl") as f:
        for i, l in enumerate(f):
            if i >= n: break
            rows.append(json.loads(l))
    idx = [r["i"] for r in rows]
    want, acts, row0 = set(idx), {}, 0
    pf = pq.ParquetFile("/vol/data/prose_L42_500k.parquet")
    for bt in pf.iter_batches(batch_size=4096, columns=["activation_vector"]):
        A = np.asarray(bt.column("activation_vector").flatten().to_numpy(zero_copy_only=False),
                       dtype="float32").reshape(-1, 5120)
        for j in range(A.shape[0]):
            if row0 + j in want: acts[row0 + j] = A[j]
        row0 += A.shape[0]
        if len(acts) == len(want): break
    ACT = torch.from_numpy(np.stack([acts[i] for i in idx])).to("cuda")
    TG = F.normalize((ACT @ J.T) - mu, dim=-1)

    lab = json.load(open("/vol/pg_dict/labels.json"))
    Vf = np.load("/vol/pg_dict/vectors.npy", mmap_mode="r")
    nt = np.load("/vol/pg_dict/n_tokens.npy")
    sel = np.nonzero(nt <= 12)[0]
    pos = {int(v): i for i, v in enumerate(sel)}          # bank row -> filtered index
    lab_f = [lab[i] for i in sel]

    def nnls(B, t):
        k = B.shape[0]; G = B @ B.T; c = B @ t; bc = 0.0
        for r in range(1, k + 1):
            for sup in itertools.combinations(range(k), r):
                ii = torch.tensor(sup, device=B.device)
                Gs = G[ii][:, ii] + 1e-6 * torch.eye(r, device=B.device)
                try: w = torch.linalg.solve(Gs, c[ii])
                except Exception: continue
                if bool((w < -1e-8).any()): continue
                rec = w.clamp(min=0) @ B[ii]; rn = rec.norm()
                if float(rn) <= 1e-8: continue
                bc = max(bc, float((rec @ t) / rn))
        return bc

    # atom texts per row, weight-ordered (all_atoms from the k=12 run)
    per = [[a for _, a in sorted(zip(r["weights"], r["all_atoms"]), key=lambda x: -x[0])]
           for r in rows]
    uniq = sorted({a for row in per for a in row})
    ui = {a: i for i, a in enumerate(uniq)}
    P = pred(uniq)
    Pa = F.normalize((P @ J.T) @ M.T, dim=-1)
    # measured counterpart for the same texts
    lut = {}
    for i, s in enumerate(lab_f): lut.setdefault(s, i)
    have = [a for a in uniq if a in lut]
    Ym = torch.from_numpy(np.ascontiguousarray(Vf[sel[[lut[a] for a in have]]])).to("cuda").float()
    Ya = F.normalize((Ym @ J.T) @ M.T, dim=-1)
    yi = {a: i for i, a in enumerate(have)}
    print("[n] %d activations | %d unique atoms (%d with a measured vector)"
          % (len(rows), len(uniq), len(have)), flush=True)
    print("%4s %12s %12s %10s %10s" % ("k", "measured", "AR-pred", "retained", "gap"), flush=True)
    out = {}
    for k in (1, 2, 4, 8, 12):
        cm_, cp_ = [], []
        for i, row in enumerate(rows):
            ats = [a for a in per[i][:k] if a in yi]
            if len(ats) < min(k, 1): continue
            Bm = Ya[torch.tensor([yi[a] for a in ats], device="cuda")]
            Bp = Pa[torch.tensor([ui[a] for a in ats], device="cuda")]
            cm_.append(nnls(Bm, TG[i])); cp_.append(nnls(Bp, TG[i]))
        mm, pp = float(np.mean(cm_)), float(np.mean(cp_))
        out[k] = {"measured": mm, "pred": pp, "retained": pp / mm if mm else 0.0}
        print("%4d %12.4f %12.4f %9.1f%% %10.4f" % (k, mm, pp, 100 * pp / max(mm, 1e-9), mm - pp),
              flush=True)
    json.dump(out, open("/vol/data/arity_amplification.json", "w"), indent=1)
    VOL.commit()
    print("ARITY_DONE", flush=True)
