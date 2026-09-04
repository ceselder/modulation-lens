"""Calibration test for ar_reward.score().

The NNOMP 4-atom decompositions have a known FVE of 0.635 computed from MEASURED atom vectors ->
cos 0.797. score() recomputes the same quantity through a completely independent path: the atom
TEXT -> frozen AR -> J -> affine -> exact NNLS. So:

  * true atoms should land BELOW 0.797 (the AR predicts vectors at ~0.91 cos to measured, so the
    composition degrades) but clearly above chance
  * random atoms from the same bank are the control -- these share the bank's mean and length
    statistics, so beating them is a content result, not a scale artifact
  * shuffled targets (right atoms, wrong activation) is the second control: it must collapse

If true >> shuffled ~ random, the reward measures what it claims.
"""
import modal

app = modal.App("celeste-test-ar-reward")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "numpy", "pyarrow", "peft",
                    "safetensors", "accelerate")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       # add_local_* must come LAST in the chain: Modal refuses a build step after it
       .add_local_file("/home/celeste/modlens-scalerl/rl/ar_reward.py", "/root/ar_reward.py"))


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200", cpu=8.0, memory=131072, timeout=5400)
def run(n: int = 256, k: int = 4, ar_dir: str = "/vol/ar_l42_text2vec"):
    import glob, json, random
    import numpy as np, torch, pyarrow.parquet as pq
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import sys
    sys.path.insert(0, "/root")
    from ar_reward import ARReward

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    m, info = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    assert not [q for q in info.get("missing_keys", []) if "lora" not in q], "weights missing"
    # stand in for the policy: any adapter, so set_adapter has something to return to
    actor = PeftModel.from_pretrained(m, ar_dir, adapter_name="policy")
    jp = glob.glob("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/*/"
                   "qwen3.6-27b/j-lens/lens.pt")[0]
    R = ARReward(ar_dir, jp, "/vol/data/affine_M_jspace.npy", read_layer=42)
    print("[ar] adapter tensors: %d" % R.attach(actor), flush=True)

    rows = []
    with open("/vol/data/nnomp_4bullets_sft.jsonl") as f:
        for i, l in enumerate(f):
            if i >= n: break
            rows.append(json.loads(l))
    idx = [r["i"] for r in rows]
    print("[data] %d rows | reference FVE %.4f -> cos %.4f"
          % (len(rows), float(np.mean([r["fve"] for r in rows])),
             float(np.mean([r["fve"] for r in rows])) ** 0.5), flush=True)

    pf = pq.ParquetFile("/vol/data/prose_L42_500k.parquet")
    want, acts = set(idx), {}
    row0 = 0
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
    TG = F.normalize((ACT @ R.J.T) - mu, dim=-1)          # exactly the space stage2_nnomp used

    def as_text(bl): return "\n".join("* " + b for b in bl)

    true_txt = [as_text(r["bullets"][:k]) for r in rows]
    rng = random.Random(0)
    pool = [b for r in rows for b in r["bullets"]]
    rand_txt = [as_text(rng.sample(pool, k)) for _ in rows]
    perm = list(range(len(rows))); rng.shuffle(perm)
    shuf_tg = TG[torch.tensor(perm, device="cuda")]

    out = {}
    for name, txt, tg in (("true_atoms", true_txt, TG),
                          ("random_atoms", rand_txt, TG),
                          ("true_atoms_SHUFFLED_targets", true_txt, shuf_tg)):
        r = R.score(txt, tg, actor, tok, k=k, max_tok=12)
        out[name] = (float(r.mean()), float(r.std()) / max(len(r), 1) ** 0.5)
        print("  %-30s cos %.4f +- %.4f | %s" % (name, out[name][0], out[name][1],
                                                 R.last_stats), flush=True)
    ref = float(np.mean([r["fve"] for r in rows])) ** 0.5
    print("\n[verdict] reference (measured vectors) cos %.4f" % ref, flush=True)
    print("[verdict] true %.4f | random %.4f | shuffled %.4f"
          % (out["true_atoms"][0], out["random_atoms"][0],
             out["true_atoms_SHUFFLED_targets"][0]), flush=True)
    ok = (out["true_atoms"][0] > out["random_atoms"][0] + 0.05
          and out["true_atoms"][0] > out["true_atoms_SHUFFLED_targets"][0] + 0.05
          and out["true_atoms"][0] < ref + 0.05)
    print("[verdict] %s" % ("PASS -- reward tracks content, magnitude consistent with the AR's fidelity"
                            if ok else "FAIL -- inspect before wiring into RL"), flush=True)
    json.dump({k2: v for k2, v in out.items()} | {"ref_cos": ref, "ok": bool(ok)},
              open("/vol/data/ar_reward_calibration.json", "w"), indent=1)
    VOL.commit()
    print("AR_REWARD_TEST_DONE", flush=True)
