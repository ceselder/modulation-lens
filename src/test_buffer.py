#!/usr/bin/env python3
"""
Why does y-first work? Two candidate mechanisms, tested by varying ONE thing at a time.

A. CARRIER ADJACENCY. All five weak templates end on/near "{y}", so the read positions sit right
   after an identical copy of themselves -- a copy-induction setup where local copying may dominate
   and crowd out {x}. Both good templates put a buffer sentence in between.
   Test: hold a template fixed, vary ONLY the buffer length after the final {y}.

B. BINDING. Both winners tie {x} grammatically to the act of writing ("while writing the following
   phrase", "as you write it, keep {x} in mind"), whereas the worst template isolates it
   ("Think about {x}."). constraint-strict has a long buffer AND is worst, so A alone cannot be it.
   Test: same buffer, {x} bound vs isolated.
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

N, NCAR = 20, 2
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
CARS = C.CARRIERS_RECOVERED[:NCAR]

BUF = {"buf0 (adjacent)": "",
       "buf1 short": " Output only that.",
       "buf2 medium": " Output only the sentence and nothing else.",
       "buf3 long": " Write only the last phrase. Don't respond with anything else.",
       "buf4 very long": " Write only the last phrase. Don't respond with anything else. "
                         "No preamble, no explanation, no commentary of any kind."}
# A: adjacency sweep on a WEAK base (x bound to the writing, so only the buffer varies)
BASE_BOUND = 'Think deeply about "{x}" while replying with nothing but "{y}".'
# B: same buffers, but {x} ISOLATED in its own sentence instead of bound
BASE_ISOL = 'Think about "{x}".\n\nReply with nothing but "{y}".'
CAND = {}
for bn, b in BUF.items():
    CAND["bound  | " + bn] = BASE_BOUND + b
    CAND["isolat | " + bn] = BASE_ISOL + b

V = []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=4096, columns=["activation_vector"]):
    V.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in V) >= 20000:
        break
AMU = torch.from_numpy(np.concatenate(V)[:20000]).mean(0).to(dev) @ J.T
d = json.load(open("/workspace/inv/results/blogpost_readouts.json"))
k_ = lambda r: (r["para"], r["i"])
g350 = {k_(r): r for r in d["runs"]["rl350"]["rows"]}
keys = list(g350)[:N]
paras = [x.strip() for x in open("/workspace/inv/data/blogpost.txt").read().split("\n\n")
         if x.strip()]
CPRE, CPOST = C.chat_wrap_ids(tok)
TGT = {}
with torch.no_grad():
    for k in keys:
        pi, ti = k
        ids = tok(paras[pi], add_special_tokens=False, truncation=True, max_length=256).input_ids
        ti = min(ti, len(ids) - 1)
        model(input_ids=torch.tensor([CPRE + ids + CPOST], device=dev))
        H = L42["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
        t = (H[ti] @ J.T) - AMU
        TGT[k] = t / t.norm().clamp(min=1e-8)
PRE = "not pizza related wording but related"
CORE = {k: (g350[k]["phrase"].replace(PRE, "").strip() or "the") for k in keys}
FULL = {k: g350[k]["phrase"] for k in keys}
import random
rr = random.Random(0)
words = ("policy river garden engine harbour lantern meadow cipher tunnel orchard beacon "
         "quarry saddle thistle vellum wharf pigment rafter cistern bramble").split()
FILL = [" ".join(rr.choice(words) for _ in range(rr.randint(3, 9))) for _ in range(40)]
ALL = sorted(set(CORE.values()) | set(FULL.values()) | set(FILL))
print("[b] %d variants x %d carriers | %d positions" % (len(CAND), NCAR, len(keys)), flush=True)

rows = []
for name, tpl in CAND.items():
    G = C.Grid(tok, [tpl], CARS, 42, J, dev)
    bm, bx, pm = [], [], []
    with torch.no_grad():
        for ci in range(NCAR):
            rv = G.read(model, ALL, L42, carrier=ci, max_tok=32)
            pmu = torch.stack([rv[f] for f in FILL]).mean(0)
            u = {s: (rv[s] - pmu) / (rv[s] - pmu).norm().clamp(min=1e-8) for s in ALL}
            for k in keys:
                bm.append(float(u[CORE[k]] @ TGT[k]))
                pm.append(float(u[FULL[k]] @ TGT[k]))
                bx += [float(u[CORE[k]] @ TGT[k2]) for k2 in keys[:8] if k2 != k]
    r = {"name": name, "bare": float(np.mean(bm)), "mis": float(np.mean(bx)),
         "pizza": float(np.mean(pm))}
    r["bonus"] = r["pizza"] - r["bare"]
    r["disc"] = r["bare"] - r["mis"]
    rows.append(r)
    print("  %-24s bare %.4f  disc %.4f  bonus %+.4f" % (name, r["bare"], r["disc"], r["bonus"]),
          flush=True)

print("\n=== A. does buffer length alone fix it?  ({x} BOUND to the writing) ===")
for r in [x for x in rows if x["name"].startswith("bound")]:
    print("  %-24s bare %.4f  disc %.4f  bonus %+.4f"
          % (r["name"].split("| ")[1], r["bare"], r["disc"], r["bonus"]))
print("\n=== B. same buffers, {x} ISOLATED in its own sentence ===")
for r in [x for x in rows if x["name"].startswith("isolat")]:
    print("  %-24s bare %.4f  disc %.4f  bonus %+.4f"
          % (r["name"].split("| ")[1], r["bare"], r["disc"], r["bonus"]))
bb = [x["bare"] for x in rows if x["name"].startswith("bound")]
bi = [x["bare"] for x in rows if x["name"].startswith("isolat")]
print("\n  bound  : bare %.4f -> %.4f as the buffer grows" % (bb[0], bb[-1]))
print("  isolated: bare %.4f -> %.4f" % (bi[0], bi[-1]))
print("  binding effect (mean bound - mean isolated) = %+.4f" % (np.mean(bb) - np.mean(bi)))
json.dump(rows, open("/workspace/inv/results/buffer.json", "w"), indent=1)
print("\nBUFFER_DONE", flush=True)
