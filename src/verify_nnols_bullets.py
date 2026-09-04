#!/usr/bin/env python3
"""What do the mined bullets actually READ BACK to?

The dictionary miner scores its picks geometrically, atom-vector against activation. But the SFT
target is the atom's LABEL TEXT, and at RL time that text is re-read through the 36-cell grid and
PMU-centred -- a different vector from the stored atom. Two things can therefore erode the mined
number before RL ever sees it: the 10-token truncation, and the atom-vector-vs-grid-read gap. This
measures the composition the trainer's reward will actually assign to the mined text, so we know
whether the warm start is worth an SFT before paying for one.

Prints the mined-vs-real gap per source file, so two warm starts can be compared directly.
"""
import argparse, json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

p = argparse.ArgumentParser()
p.add_argument("--bullets", required=True, help="comma-separated mined jsonl files to compare")
p.add_argument("--data", default="/workspace/inv/data/prose_L42.parquet")
p.add_argument("--pmu", default="/workspace/inv/ckpts/rl_v2_plain/pmu_db4a6b8ee6.npy")
p.add_argument("--n-rows", type=int, default=256)
p.add_argument("--n-pool", type=int, default=60000)
p.add_argument("--min-words", type=int, default=4)
p.add_argument("--n-carriers", type=int, default=6)
p.add_argument("--max-tok", type=int, default=10, help="grid read length cap; match --bullet-max-tok")
A = p.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = model.model
L42 = {}
inner.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[: A.n_carriers], 42, J, dev)
PMU = torch.from_numpy(np.load(A.pmu)).to(dev).float()

acc_v, acc_l = [], []
for b in pq.ParquetFile(A.data).iter_batches(batch_size=4096,
                                             columns=["activation_vector", "label"]):
    d = b.to_pydict()
    acc_v.append(np.asarray(d["activation_vector"], dtype="float32")); acc_l += d["label"]
    if sum(len(x) for x in acc_v) >= A.n_pool:
        break
V = np.concatenate(acc_v)
keep = [i for i, l in enumerate(acc_l) if len(l.split()) >= A.min_words][: A.n_pool]
AMU = torch.from_numpy(V[keep].mean(0)).to(dev) @ J.T
print("[i] pool %d -> %d filtered | |AMU| %.2f |PMU| %.2f"
      % (V.shape[0], len(keep), float(AMU.norm()), float(PMU.norm())), flush=True)


def nnls_small(Bm, t):
    n = Bm.shape[0]
    bw, br = None, float("inf")
    for mask in range(1, 1 << n):
        idx = [k for k in range(n) if (mask >> k) & 1]
        S = Bm[idx].T
        sol = torch.linalg.lstsq(S, t.unsqueeze(1)).solution.squeeze(1)
        if bool((sol < -1e-8).any()):
            continue
        r = float((t - S @ sol).norm())
        if r < br:
            br = r
            w = torch.zeros(n, device=Bm.device, dtype=Bm.dtype)
            w[torch.tensor(idx, device=Bm.device)] = sol
            bw = w
    return bw if bw is not None else torch.zeros(n, device=Bm.device, dtype=Bm.dtype)


for path in A.bullets.split(","):
    path = path.strip()
    if not path:
        continue
    rows = [json.loads(l) for l in open(path) if l.strip()][: A.n_rows]
    real, single, mined, nz = [], [], [], []
    with torch.no_grad():
        for r in rows:
            bl = [b for b in r["bullets"] if b.strip()]
            if len(bl) < 2:
                continue
            cv = GRID.read_all(model, bl, L42, max_tok=A.max_tok)
            M = torch.stack([cv[b] - PMU for b in bl])
            t = (torch.from_numpy(V[int(r["i"])]).to(dev) @ J.T) - AMU
            t = t / t.norm().clamp(min=1e-8)
            w = nnls_small(M, t)
            rec = w @ M
            real.append(float((rec @ t) / rec.norm().clamp(min=1e-8)))
            single.append(max(float(M[k] @ t / M[k].norm().clamp(min=1e-8))
                              for k in range(M.shape[0])))
            mined.append(float(r.get("compose_cos", float("nan"))))
            nz.append(int((w > 1e-6).sum()))
    print("\n%s  (%d rows)" % (path, len(real)))
    print("  mined geometric compose_cos   %.4f" % float(np.nanmean(mined)))
    print("  REAL grid-read compose_cos    %.4f   <- what RL will reward" % float(np.mean(real)))
    print("  REAL best single bullet       %.4f" % float(np.mean(single)))
    print("  composition gain over single  %+.4f" % float(np.mean(real) - np.mean(single)))
    print("  bullets NNLS actually uses    %.2f of %d" % (float(np.mean(nz)),
                                                          max(len(r["bullets"]) for r in rows)))
print("\nVERIFY_DONE", flush=True)
