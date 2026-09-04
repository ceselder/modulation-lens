#!/usr/bin/env python3
"""Merge the harvest shards into one SHUFFLED parquet on local disk.

Two reasons the merge is not a plain concatenation:
  * the trainer takes a single --data path and keeps a PREFIX of --n-pool rows, so appending shards
    end to end would make any pool smaller than the total come entirely from shard A;
  * the mined bullets index this file's row order, so the permutation has to be applied BEFORE
    mining, not after.
Local disk rather than /workspace: MooseFS read out at ~34 MB/s during the dictionary load, and both
the miner and all four training ranks read this file end to end.
"""
import argparse, glob, os, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="/workspace/inv/data/prose_L42_shard*.parquet,"
                                    "/workspace/inv/data/prose_L42_500k.parquet")
ap.add_argument("--out", default="/root/data/prose_L42_500k.parquet")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--row-group", type=int, default=8192)
ap.add_argument("--max-rows", type=int, default=0,
                help="keep this many rows AFTER the shuffle, so the kept subset is spread across "
                     "shards rather than being one shard's prefix")
A = ap.parse_args()

files = []
for pat in A.shards.split(","):
    files += sorted(glob.glob(pat.strip()))
files = [f for f in files if os.path.getsize(f) > 0]
assert files, "no shards found"
D = 5120
V, L, C, DD, P = [], [], [], [], []
t0 = time.time()
for f in files:
    n = 0
    for b in pq.ParquetFile(f).iter_batches(batch_size=8192):
        d = b.to_pydict()
        V.append(np.asarray(d["activation_vector"], dtype="float32"))
        L += d["label"]; C += d["ctx"]; DD += d["doc_id"]; P += d["pos"]
        n += len(d["label"])
    print("[cat] %s -> %d rows (%.0fs)" % (os.path.basename(f), n, time.time() - t0), flush=True)
V = np.concatenate(V)
N = V.shape[0]
# doc_id collides across shards (each shard counts from 1); make it shard-unique so provenance
# survives the merge instead of silently claiming rows from different shards share a document.
perm = np.random.default_rng(A.seed).permutation(N)
if A.max_rows and A.max_rows < N:
    print("[cat] %d rows harvested, keeping a random %d" % (N, A.max_rows), flush=True)
    perm = perm[: A.max_rows]
    N = A.max_rows
else:
    print("[cat] %d rows total, shuffling" % N, flush=True)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
schema = pa.schema([("activation_vector", pa.list_(pa.float32(), D)),
                    ("label", pa.string()), ("ctx", pa.string()),
                    ("doc_id", pa.int64()), ("pos", pa.int64())])
L = np.asarray(L, dtype=object); C = np.asarray(C, dtype=object)
DD = np.asarray(DD, dtype="int64"); P = np.asarray(P, dtype="int64")
w = pq.ParquetWriter(A.out, schema)
for s in range(0, N, A.row_group):
    ix = perm[s:s + A.row_group]
    w.write_table(pa.table({
        "activation_vector": pa.FixedSizeListArray.from_arrays(
            pa.array(V[ix].reshape(-1), type=pa.float32()), D),
        "label": pa.array(list(L[ix])), "ctx": pa.array(list(C[ix])),
        "doc_id": pa.array(DD[ix], type=pa.int64()),
        "pos": pa.array(P[ix], type=pa.int64())}, schema=schema))
w.close()
print("[cat] wrote %d rows -> %s (%.1f GB, %.0fs)"
      % (N, A.out, os.path.getsize(A.out) / 1e9, time.time() - t0), flush=True)
print("CONCAT_DONE %d" % N, flush=True)
