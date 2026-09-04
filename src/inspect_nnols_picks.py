#!/usr/bin/env python3
"""Are the K=16 NNOLS picks semantically coherent, or geometric filler?

Reconstruction cosine is blind to label relevance -- NNOLS scored 0.184 on workspace-bench while
reconstructing fine. So print, for real blogpost probe activations, the 16 atoms greedy NNOLS
selects WITH their non-negative weights and the running reconstruction, next to the text the
activation was actually read from. That shows directly whether a Claude-condensation step would
have signal to work with, and how much of the reconstruction the top few atoms carry.
"""
import glob, json, sys
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C
dev, K = "cuda", 16
J = C.load_jlens(42, dev)
labs, vecs = [], []
for sh in sorted(glob.glob("/workspace/thinkies/v3/thinkies_v3-*.parquet")):
    for b in pq.ParquetFile(sh).iter_batches(batch_size=16384, columns=["label", "vector"]):
        l = b.column("label").to_pylist(); labs += l
        vecs.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                               dtype="float16").reshape(len(l), -1))
AJ = (torch.from_numpy(np.concatenate(vecs)).to(dev, torch.float16).float() @ J.T).half()
del vecs
AJn = AJ.float().norm(dim=1).clamp(min=1e-6)
print("[i] %d atoms" % AJ.shape[0], flush=True)

acc, n = np.zeros(5120, dtype="float64"), 0
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=8192, columns=["activation_vector"]):
    a = np.asarray(b.to_pydict()["activation_vector"], dtype="float32")
    acc += a.sum(0); n += len(a)
    if n >= 40000:
        break
mu = torch.from_numpy((acc / n).astype("float32")).to(dev)
ACT = torch.from_numpy(np.load("/workspace/inv/data/holdout_blogpost.npy").astype("float32"))
META = [json.loads(l) for l in open("/workspace/inv/data/holdout_blogpost.jsonl")]

for i in (0, 1, 4, 7):
    t = (ACT[i].to(dev) - mu) @ J.T
    t = t / t.norm().clamp(min=1e-8)
    resid, chosen, hist = t.clone(), [], []
    for step in range(K):
        c = (AJ.half() @ resid.half()).float() / AJn
        if chosen:
            c[torch.tensor(chosen, device=dev)] = -1e9
        chosen.append(int(c.argmax()))
        S = AJ[torch.tensor(chosen, device=dev)].float().T
        w = torch.linalg.lstsq(S, t.unsqueeze(1)).solution.squeeze(1).clamp(min=0)
        rec = S @ w
        hist.append(float((rec @ t) / rec.norm().clamp(min=1e-8)))
        resid = t - rec
    m = META[i] if i < len(META) else {}
    print("\n" + "=" * 96)
    print("PROBE %d  token %r   context: ...%s" % (i, m.get("mark", "?"), m.get("ctx", "")[-72:]))
    print("=" * 96)
    for r, (j, cum) in enumerate(zip(chosen, hist)):
        print("  %2d. w=%6.3f  cum-cos %.3f   %s" % (r + 1, float(w[r]), cum, labs[j][:66]))
print("\nINSPECT_DONE", flush=True)
