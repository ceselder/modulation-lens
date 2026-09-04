#!/usr/bin/env python3
"""
Does the TARGET side need its mean subtracted? Grid v2, plain cosine.

Argument for dropping it: if both sides live in the same J-space, cosine might handle the offset.
Argument against: the candidate side subtracts PMU (the grid's own mean) while the target would keep
its full offset, so the two sides are NOT in the same frame -- and every candidate then competes to
match a component shared across all targets, which is exactly what pays a generic phrase.

`bonus` (what a boilerplate wrapper still buys) is our established proxy for hackability, so this
predicts template-collapse without a training run.

2x2: {target centred, target raw} x {candidate centred, candidate raw}.
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

LO, HI = 40, 76
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, 42, J, dev)   # full v2 grid
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
keys = list(g350)[LO:HI]
paras = [x.strip() for x in open("/workspace/inv/data/blogpost.txt").read().split("\n\n")
         if x.strip()]
CPRE, CPOST = C.chat_wrap_ids(tok)
RAWT = {}
with torch.no_grad():
    for k in keys:
        pi, ti = k
        ids = tok(paras[pi], add_special_tokens=False, truncation=True, max_length=256).input_ids
        ti = min(ti, len(ids) - 1)
        model(input_ids=torch.tensor([CPRE + ids + CPOST], device=dev))
        H = L42["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
        RAWT[k] = (H[ti] @ J.T)
PRE = "not pizza related wording but related"
CORE = {k: (g350[k]["phrase"].replace(PRE, "").strip() or "the") for k in keys}
FULL = {k: g350[k]["phrase"] for k in keys}
import random
rr = random.Random(0)
words = ("policy river garden engine harbour lantern meadow cipher tunnel orchard beacon quarry "
         "saddle thistle vellum wharf pigment rafter cistern bramble").split()
FILL = [" ".join(rr.choice(words) for _ in range(rr.randint(3, 9))) for _ in range(24)]
ALL = sorted(set(CORE.values()) | set(FULL.values()) | set(FILL))
print("[c] full v2 grid: %d tpl x %d car = %d cells | %d positions"
      % (GRID.n_tpl, GRID.n_car, GRID.n_tpl * GRID.n_car, len(keys)), flush=True)
with torch.no_grad():
    RV = GRID.read_all(model, ALL, L42, max_tok=32)
PMU = torch.stack([RV[f] for f in FILL]).mean(0)
print("[c] |PMU| %.2f  |AMU| %.2f" % (float(PMU.norm()), float(AMU.norm())), flush=True)


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


print("\n%-34s %8s %8s %9s %9s" % ("configuration", "bare", "disc", "bonus", "mis"))
rows = []
for tname, tc in (("target centred", True), ("target RAW", False)):
    for cname, cc in (("cand centred", True), ("cand RAW", False)):
        bm, bx, pm = [], [], []
        for k in keys:
            t = RAWT[k] - AMU if tc else RAWT[k]
            a = RV[CORE[k]] - PMU if cc else RV[CORE[k]]
            f = RV[FULL[k]] - PMU if cc else RV[FULL[k]]
            bm.append(cos(a, t)); pm.append(cos(f, t))
            bx += [cos(a, (RAWT[k2] - AMU) if tc else RAWT[k2]) for k2 in keys[:8] if k2 != k]
        r = {"cfg": "%s / %s" % (tname, cname), "bare": float(np.mean(bm)),
             "mis": float(np.mean(bx)), "pizza": float(np.mean(pm))}
        r["disc"] = r["bare"] - r["mis"]
        r["bonus"] = r["pizza"] - r["bare"]
        rows.append(r)
        print("%-34s %8.4f %8.4f %+9.4f %+9.4f"
              % (r["cfg"], r["bare"], r["disc"], r["bonus"], r["mis"]), flush=True)
base = rows[0]
print("\n=== vs the current setup (target centred / cand centred) ===")
for r in rows[1:]:
    worse = r["bonus"] > base["bonus"]
    print("  %-32s disc %+.4f   bonus %+.4f  -> %s"
          % (r["cfg"], r["disc"] - base["disc"], r["bonus"],
             "MORE hackable" if worse else "less hackable"))
print("\n  prediction was: dropping target centring leaves a large shared component in every")
print("  target, so a generic phrase pays -> bonus should RISE and mis should RISE.")
json.dump(rows, open("/workspace/inv/results/centering.json", "w"), indent=1)
print("\nCENTERING_DONE", flush=True)
