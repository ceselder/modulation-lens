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
app = modal.App("celeste-diag-ar-space")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "numpy", "peft", "safetensors",
                    "accelerate")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200", cpu=8.0, memory=131072, timeout=5400)
def run(n: int = 4096, ar_dir: str = "/vol/ar_l42_text2vec", max_tokens: int = 12):
    import glob, json
    import numpy as np, torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    m, info = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    assert not [q for q in info.get("missing_keys", []) if "lora" not in q]
    base = m.model
    base.layers = base.layers[:43]
    del m.lm_head
    model = PeftModel.from_pretrained(base, ar_dir).eval()
    hd = torch.load(ar_dir + "/head.pt", map_location="cuda")
    head = torch.nn.Linear(5120, 5120, bias=True).to("cuda", torch.float32)
    head.load_state_dict(hd["head"]); head.eval()
    HK = {}
    model.base_model.model.layers[42].register_forward_hook(
        lambda mm, i, o: HK.__setitem__("h", o[0] if isinstance(o, tuple) else o))

    labels = json.load(open("/vol/pg_dict/labels.json"))
    V = np.load("/vol/pg_dict/vectors.npy", mmap_mode="r")
    ev = np.load("/vol/ar_stage1/split_eval_idx.npy")
    rng = np.random.default_rng(0)
    ev = ev[rng.permutation(len(ev))][:n]           # random, not the sorted prefix
    ev = np.sort(ev)

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

    P = pred([labels[i] for i in ev])
    Y = torch.from_numpy(np.ascontiguousarray(V[ev])).to("cuda", torch.float32)
    jp = glob.glob("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/*/"
                   "qwen3.6-27b/j-lens/lens.pt")[0]
    J = torch.load(jp, map_location="cpu", weights_only=False)["J"][42].to("cuda").float()
    M = torch.from_numpy(np.load("/vol/data/affine_M_jspace.npy")).to("cuda").float()
    W = torch.from_numpy(np.load("/vol/ar_stage1/ridge_W.npy")).to("cuda").float()

    def cm(A, B): return float((F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(1).mean())

    print("[n] %d held-out atoms (random subset)" % len(ev), flush=True)
    rows = [("AR", P)]
    # ridge features: mean input embedding, same as ar_data_ridge
    E = None
    try:
        E = m.get_input_embeddings().weight.detach().float()
    except Exception:
        pass
    if E is not None:
        Xb = torch.zeros((len(ev), 5120), device="cuda")
        for a in range(0, len(ev), 20000):
            ids = tok([labels[i] for i in ev[a:a+20000]], add_special_tokens=False)["input_ids"]
            for j, s in enumerate(ids):
                if s: Xb[a+j] = E[torch.tensor(s, device="cuda")].mean(0)
        rows.append(("ridge", Xb @ W))
    print("%-7s %10s %10s %10s" % ("pred", "cos_raw", "cos_J", "cos_JM"), flush=True)
    res = {}
    for name, Pp in rows:
        cr = cm(Pp, Y)
        cj = cm(Pp @ J.T, Y @ J.T)
        cjm = cm((Pp @ J.T) @ M.T, (Y @ J.T) @ M.T)
        res[name] = {"cos_raw": cr, "cos_J": cj, "cos_JM": cjm}
        print("%-7s %10.4f %10.4f %10.4f" % (name, cr, cj, cjm), flush=True)
    print("\n[attribution] AR: raw %.4f -> after J %.4f (%+.4f) -> after affine %.4f (%+.4f)"
          % (res["AR"]["cos_raw"], res["AR"]["cos_J"],
             res["AR"]["cos_J"] - res["AR"]["cos_raw"], res["AR"]["cos_JM"],
             res["AR"]["cos_JM"] - res["AR"]["cos_J"]), flush=True)
    json.dump(res, open("/vol/data/ar_space_attribution.json", "w"), indent=1)
    VOL.commit()
    print("DIAG_DONE", flush=True)
