#!/usr/bin/env python3
"""How well does NNOLS over the thinkies dict reconstruct a REAL activation, as a function of K?

This gates the proposed warm start: SFT targets built from a K-atom NNOLS decomposition are only
worth training on if that decomposition actually explains the activation. Known bad news at K=1:
the nearest single atom sits at cos 0.375 in J-space (44% of the atom-to-atom neighbour cosine), so
the question is whether K=8/16/32 closes the gap.

Greedy non-negative matching pursuit (pick the atom best correlated with the residual, refit the
whole active set by NNLS, recurse), scored in J-space where reachability measured best
(jspace 44.6% > raw 22.1% > whitened 15.8%).
"""
import argparse, glob, sys
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

p = argparse.ArgumentParser()
p.add_argument("--max-atoms", type=int, default=1600000)
p.add_argument("--n-targets", type=int, default=32)
p.add_argument("--ks", default="1,2,4,8,16,32")
A = p.parse_args()
dev = "cuda"
J = C.load_jlens(42, dev)

labs, vecs = [], []
for sh in sorted(glob.glob("/workspace/thinkies/v3/thinkies_v3-*.parquet")):
    for b in pq.ParquetFile(sh).iter_batches(batch_size=16384, columns=["label", "vector"]):
        l = b.column("label").to_pylist(); labs += l
        vecs.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                               dtype="float16").reshape(len(l), -1))
        if len(labs) >= A.max_atoms:
            break
    if len(labs) >= A.max_atoms:
        break
A_raw = torch.from_numpy(np.concatenate(vecs)).to(dev, torch.float16)
del vecs
AJ = (A_raw.float() @ J.T).half()          # atoms in J-space
del A_raw
AJn = AJ.float().norm(dim=1).clamp(min=1e-6)
print("[k] dict %d atoms in J-space" % AJ.shape[0], flush=True)

acc, n, rows = np.zeros(5120, dtype="float64"), 0, []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=8192, columns=["activation_vector"]):
    a = np.asarray(b.to_pydict()["activation_vector"], dtype="float32")
    acc += a.sum(0); n += len(a); rows.append(a)
    if n >= 40000:
        break
mu = torch.from_numpy((acc / n).astype("float32")).to(dev)
P = torch.from_numpy(np.concatenate(rows)[: A.n_targets].astype("float32")).to(dev) - mu
PJ = P @ J.T
PJ = PJ / PJ.norm(dim=1, keepdim=True).clamp(min=1e-8)
KS = [int(x) for x in A.ks.split(",")]

res = {k: [] for k in KS}
picks_at_max = []
for i in range(PJ.shape[0]):
    t = PJ[i]
    resid = t.clone()
    chosen = []
    for step in range(max(KS)):
        c = (AJ.half() @ resid.half()).float() / AJn
        if chosen:
            c[torch.tensor(chosen, device=dev)] = -1e9
        chosen.append(int(c.argmax()))
        S = AJ[torch.tensor(chosen, device=dev)].float().T          # [d,k]
        w = torch.linalg.lstsq(S, t.unsqueeze(1)).solution.squeeze(1).clamp(min=0)
        rec = S @ w
        resid = t - rec
        k = step + 1
        if k in res:
            res[k].append(float((rec @ t) / rec.norm().clamp(min=1e-8)))
    picks_at_max.append(list(chosen))
    if i == 0:
        print("[k] target 0 first 8 picks: %s" % [labs[j][:26] for j in chosen[:8]], flush=True)

print("\n  K   mean cos   median   |  what it means")
for k in KS:
    v = np.array(res[k])
    print("  %-3d %.4f     %.4f" % (k, v.mean(), np.median(v)))
print("\n  atom-to-atom neighbour cosine in J-space was 0.84; single best atom 0.375.")
print("  A warm start needs the K-atom decomposition to EXPLAIN the activation. Judge viability")
print("  against the four-bullet lens's own reconstruction, which reaches ~0.51 with 4 bullets.")
print("NNOLS_K_DONE", flush=True)
