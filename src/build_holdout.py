#!/usr/bin/env python3
"""Build a properly-sized held-out probe from the harvest rows that training never saw.

The in-run probe is 32 activations. Per-item composition cosine spreads by roughly 0.15, so the
standard error of that mean is ~0.027 and the probe cannot resolve anything smaller than ~0.05 --
which is most of the trajectory. Every movement after step 25 sat inside it.

No new harvesting is needed: concat_shards.py kept a RANDOM 500,000 of the 860,293 harvested rows
under a fixed seed, so the 360,293 dropped rows are same-distribution and untouched by training.
Recompute that permutation and take the complement.
"""
import argparse, json
import numpy as np
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", default=[
    "/workspace/inv/data/prose_L42_shardB.parquet",
    "/workspace/inv/data/prose_L42_shardC.parquet",
    "/workspace/inv/data/prose_L42_500k.parquet"],
    help="IN THE SAME ORDER concat_shards.py read them, or the permutation will not line up")
ap.add_argument("--kept", type=int, default=500000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--out-npy", default="/workspace/inv/data/holdout_fresh.npy")
ap.add_argument("--out-meta", default="/workspace/inv/data/holdout_fresh.jsonl")
A = ap.parse_args()

counts = [pq.ParquetFile(f).metadata.num_rows for f in A.files]
N = sum(counts)
print("[i] concatenated rows: %s = %d" % (" + ".join(map(str, counts)), N), flush=True)
perm = np.random.default_rng(A.seed).permutation(N)
dropped = perm[A.kept:]
print("[i] never used by training: %d" % len(dropped), flush=True)
assert len(dropped) >= A.n, "not enough unused rows"
# a deterministic, distribution-spanning sample of the unused rows
pick = np.sort(np.random.default_rng(12345).choice(dropped, size=A.n, replace=False))
want = set(int(x) for x in pick)

V = np.zeros((A.n, 5120), dtype="float32")
meta = [None] * A.n
order = {int(g): k for k, g in enumerate(pick)}
base = 0
got = 0
for f, cnt in zip(A.files, counts):
    lo, hi = base, base + cnt
    if not any(lo <= g < hi for g in want):
        base = hi
        continue
    off = 0
    for b in pq.ParquetFile(f).iter_batches(batch_size=8192,
                                            columns=["activation_vector", "label", "ctx"]):
        d = b.to_pydict()
        m = len(d["label"])
        for j in range(m):
            g = base + off + j
            if g in want:
                k = order[g]
                V[k] = np.asarray(d["activation_vector"][j], dtype="float32")
                meta[k] = {"mark": (d["label"][j] or " ")[-1], "ctx": d["ctx"][j],
                           "label": d["label"][j], "global_row": g}
                got += 1
        off += m
    base = hi
    print("[i] %s -> %d/%d collected" % (f.split("/")[-1], got, A.n), flush=True)
assert got == A.n and all(meta), "collected %d of %d" % (got, A.n)
np.save(A.out_npy, V)
with open(A.out_meta, "w") as g:
    for r in meta:
        g.write(json.dumps(r) + "\n")
print("[done] %d held-out activations -> %s (+ meta)" % (A.n, A.out_npy), flush=True)
print("HOLDOUT_DONE", flush=True)
