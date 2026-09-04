"""Why does ar_reward.score() give 0.329 where the direct path gives 0.750 on the same atoms?

Three hypotheses tested and killed already: J-conditioning (AR keeps 0.93 through J and the
affine), composition amplification (retention is flat 95% at k=1,2,4), leading-space stripping
(zero atoms in the bank carry leading or trailing whitespace). So bisect instead of theorise:
run BOTH paths in one process on the SAME rows and diff the intermediates.

  A  direct     : atom strings -> AR -> J -> M -> exact NNLS            (the diagnostic's path)
  B  score()    : '* '-joined text -> split_bullets -> embed -> NNLS     (the reward's path)
  B1 score() but with split_bullets bypassed (atoms passed as-is)
  B2 direct path but using ARReward.embed() instead of a local pred()

A vs B2 isolates embed(); B vs B1 isolates split_bullets.
"""
import modal
app = modal.App("celeste-diag-bisect")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "numpy", "pyarrow", "peft",
                    "safetensors", "accelerate")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       .add_local_file("/home/celeste/modlens-scalerl/rl/ar_reward.py", "/root/ar_reward.py"))


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200", cpu=8.0, memory=131072, timeout=5400)
def run(n: int = 256, k: int = 4, ar_dir: str = "/vol/ar_l42_text2vec", truncate: int = 1):
    import glob, itertools, json, sys
    import numpy as np, torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import pyarrow.parquet as pq
    sys.path.insert(0, "/root")
    import ar_reward as ARR

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    m, info = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    assert not [q for q in info.get("missing_keys", []) if "lora" not in q]
    if truncate:
        base = m.model; base.layers = base.layers[:43]; del m.lm_head
        actor = PeftModel.from_pretrained(base, ar_dir, adapter_name="policy").eval()
    else:
        actor = PeftModel.from_pretrained(m, ar_dir, adapter_name="policy").eval()
    print("[cfg] truncate=%d | wrapped=%s" % (truncate, type(actor.base_model.model).__name__),
          flush=True)

    jp = glob.glob("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/*/"
                   "qwen3.6-27b/j-lens/lens.pt")[0]
    R = ARR.ARReward(ar_dir, jp, "/vol/data/affine_M_jspace.npy", read_layer=42)
    R.attach(actor)

    rows = []
    with open("/vol/data/nnomp_4bullets_sft.jsonl") as f:
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
    z = np.load("/vol/data/natural_whitener_jspace.npz")
    mu = torch.tensor(z["mu"], device="cuda").float()
    TG = F.normalize((ACT @ R.J.T) - mu, dim=-1)
    per = [r["bullets"][:k] for r in rows]

    def nnls_local(B, t):
        kk = B.shape[0]; G = B @ B.T; c = B @ t; bc = 0.0
        for r_ in range(1, kk + 1):
            for sup in itertools.combinations(range(kk), r_):
                ii = torch.tensor(sup, device=B.device)
                Gs = G[ii][:, ii] + 1e-6 * torch.eye(r_, device=B.device)
                try: w = torch.linalg.solve(Gs, c[ii])
                except Exception: continue
                if bool((w < -1e-8).any()): continue
                rec = w.clamp(min=0) @ B[ii]; rn = rec.norm()
                if float(rn) <= 1e-8: continue
                bc = max(bc, float((rec @ t) / rn))
        return bc

    uniq = sorted({a for row in per for a in row})
    ui = {a: i for i, a in enumerate(uniq)}
    V_embed = R.embed(uniq, actor, tok)                      # via ARReward.embed
    res = {}

    # A / B2: direct NNLS on embed()'s vectors
    cs = [nnls_local(V_embed[torch.tensor([ui[a] for a in per[i]], device="cuda")], TG[i])
          for i in range(len(rows))]
    res["B2_direct_nnls_on_embed"] = float(np.mean(cs))

    # B: the reward as written
    txt = ["\n".join("* " + b for b in row) for row in per]
    res["B_score_as_written"] = float(R.score(txt, TG, actor, tok, k=k, max_tok=12).mean())

    # B1: score() with split_bullets bypassed
    _orig = ARR.split_bullets
    ARR.split_bullets = lambda text, kk, mt, tk: [x[2:] if x.startswith("* ") else x
                                                  for x in text.split("\n") if x.strip()]
    try:
        res["B1_score_no_splitbullets"] = float(R.score(txt, TG, actor, tok, k=k, max_tok=12).mean())
    finally:
        ARR.split_bullets = _orig

    # do the parsed strings differ from the originals at all?
    parsed = [ARR.split_bullets(t, k, 12, tok) for t in txt]
    nmis = sum(1 for i in range(len(per)) for a, b in zip(per[i], parsed[i]) if a != b)
    ntot = sum(len(x) for x in per)
    print("\n[strings] %d of %d parsed bullets differ from the original atom" % (nmis, ntot),
          flush=True)
    for i in range(len(per)):
        d = [(a, b) for a, b in zip(per[i], parsed[i]) if a != b]
        if d:
            print("   orig %r\n   got  %r" % (d[0][0], d[0][1]), flush=True)
            break
    print("\n%-32s %s" % ("path", "mean cos"), flush=True)
    for kk, v in res.items():
        print("%-32s %.4f" % (kk, v), flush=True)
    json.dump(res | {"n_string_mismatch": nmis, "n_bullets": ntot},
              open("/vol/data/bisect.json", "w"), indent=1)
    VOL.commit()
    print("BISECT_DONE", flush=True)
