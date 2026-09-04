#!/usr/bin/env python3
"""Export mined rows with their reading CONTEXT, for a Claude-authored four-bullet warm start.

The dictionary warm start transferred bullet form but not content: SFT held ppl ~28 because the
policy cannot predict WHICH atom, so it learned the style and lost the readout (probe 0.4332 ->
0.3156). The fix the evidence points to is the plan originally asked for -- let Claude write the four
bullets, using the dictionary picks as candidates rather than as targets. That needs the text the
activation was read from, which the miner does not carry, so join `ctx` back from the pool.
"""
import argparse, json
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="/root/data/prose_L42_500k.parquet")
ap.add_argument("--mined", default="/root/data/nnols4_500k.jsonl")
ap.add_argument("--out", default="/root/data/for_claude_60k.jsonl")
ap.add_argument("--n", type=int, default=60000)
A = ap.parse_args()

rows = []
with open(A.mined) as f:
    for line in f:
        if len(rows) >= A.n:
            break
        rows.append(json.loads(line))
want = {int(r["i"]): k for k, r in enumerate(rows)}
ctx = {}
seen = 0
for b in pq.ParquetFile(A.pool).iter_batches(batch_size=8192, columns=["ctx", "label"]):
    d = b.to_pydict()
    for j in range(len(d["ctx"])):
        gi = seen + j
        if gi in want:
            ctx[gi] = d["ctx"][j]
    seen += len(d["ctx"])
with open(A.out, "w") as g:
    n = 0
    for r in rows:
        c = ctx.get(int(r["i"]))
        if not c:
            continue
        g.write(json.dumps({"i": r["i"], "label": r["label"], "ctx": c,
                            "candidates": r["atom_labels_full"],
                            "weights": r["weights"],
                            "nnols_compose_cos": r["compose_cos"]}) + "\n")
        n += 1
print("wrote %d rows -> %s (ctx found for %d of %d)" % (n, A.out, len(ctx), len(rows)), flush=True)
print("EXPORT_DONE", flush=True)
