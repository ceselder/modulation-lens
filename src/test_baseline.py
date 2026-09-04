#!/usr/bin/env python3
"""
Is the contrastive "not A but B" form winning because the METRIC is a difference?

PMU is the mean over slot fillers of "thinking about X while writing the carrier" -- i.e. the state
of thinking about a generic abstract X. Subtracting it makes the candidate a deviation-from-generic;
the target is also a deviation (h - AMU). Both sides are differences, and "not A but B" is the
linguistic form of a difference. So the contrastive phrasing may be aligning with the metric's
STRUCTURE rather than describing the state better.

Evidence so far: prefix alone 0.0251, suffix alone 0.2721, full 0.3448 -- the wrapper adds ~+0.07
that is in neither part.

Baselines compared:
  PMU     mean over many slot fillers        (current -- 'relative to a generic thought')
  EMPTY   the same template with an EMPTY slot ('absolute effect of adding this phrase')
  NONE    no subtraction at all
If the wrapper's +0.07 shrinks under EMPTY, the hypothesis is confirmed and the baseline is the fix.
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

N = 24
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[:3], 42, J, dev)
V = []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=4096, columns=["activation_vector"]):
    V.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in V) >= 20000:
        break
AMU = torch.from_numpy(np.concatenate(V)[:20000]).mean(0).to(dev) @ J.T

PMU = torch.tensor(np.load("/workspace/inv/ckpts/rl/pmu.npy"), device=dev)
with torch.no_grad():
    EMPTY = GRID.read(model, ["", " "], L42, carrier=0, max_tok=4)
EMPTY = (EMPTY[""] + EMPTY[" "]) / 2
print("[b] |PMU| %.2f  |EMPTY| %.2f  cos(PMU,EMPTY) %.4f"
      % (float(PMU.norm()), float(EMPTY.norm()),
         float((PMU @ EMPTY) / (PMU.norm() * EMPTY.norm()))), flush=True)

d = json.load(open("/workspace/inv/results/blogpost_readouts.json"))
k_ = lambda r: (r["para"], r["i"])
g50 = {k_(r): r for r in d["runs"]["rl50"]["rows"]}
g350 = {k_(r): r for r in d["runs"]["rl350"]["rows"]}
keys = [k for k in g50 if k in g350][:N]
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

PRE350 = "not pizza related wording but related"
# hand-made pairs: identical content, with and without the contrastive wrapper
PAIRS = [("not searching anymore but found it", "found it at last"),
         ("not calm but excited and interrupting", "excited and interrupting"),
         ("not definitive but unsure and hedging", "unsure and hedging"),
         ("not literal travel but opportunity cost", "opportunity cost"),
         ("not enthusiasm but hesitant agreement", "hesitant agreement")]

SETS = {
    "rl50 full": [g50[k]["phrase"] for k in keys],
    "rl350 full": [g350[k]["phrase"] for k in keys],
    "rl350 suffix only": [g350[k]["phrase"].replace(PRE350, "").strip() or "the" for k in keys],
}
ALL = sorted({p for v in SETS.values() for p in v} |
             {x for ab in PAIRS for x in ab})
with torch.no_grad():
    RAW = GRID.read(model, ALL, L42, carrier=0, max_tok=24)

BASE = {"PMU (current)": PMU, "EMPTY slot": EMPTY, "NONE": torch.zeros_like(PMU)}


def sc(p, k, b):
    v = RAW[p] - b
    v = v / v.norm().clamp(min=1e-8)
    return float(v @ TGT[k])


print("\n=== phrase sets, mean matched score under each baseline ===")
print("%-22s %14s %14s %14s" % ("", "PMU (current)", "EMPTY slot", "NONE"))
res = {}
for name, ph in SETS.items():
    row = {}
    for bn, b in BASE.items():
        row[bn] = float(np.mean([sc(ph[i], keys[i], b) for i in range(len(keys))]))
    res[name] = row
    print("%-22s %14.4f %14.4f %14.4f"
          % (name, row["PMU (current)"], row["EMPTY slot"], row["NONE"]), flush=True)
print("\n=== the wrapper's free bonus:  rl350 full  minus  rl350 suffix only ===")
for bn in BASE:
    g = res["rl350 full"][bn] - res["rl350 suffix only"][bn]
    print("  %-16s %+.4f" % (bn, g))
print("\n=== matched pairs: same content, contrastive wrapper vs plain ===")
print("%-40s %-30s %9s %9s" % ("contrastive", "plain", "PMU d", "EMPTY d"))
pr = []
for a, b in PAIRS:
    da = float(np.mean([sc(a, k, BASE["PMU (current)"]) - sc(b, k, BASE["PMU (current)"])
                        for k in keys]))
    de = float(np.mean([sc(a, k, BASE["EMPTY slot"]) - sc(b, k, BASE["EMPTY slot"])
                        for k in keys]))
    pr.append((da, de))
    print("%-40s %-30s %+9.4f %+9.4f" % (a[:38], b[:28], da, de), flush=True)
print("\n  mean contrastive advantage:  PMU %+.4f   EMPTY %+.4f"
      % (float(np.mean([x[0] for x in pr])), float(np.mean([x[1] for x in pr]))))
json.dump({"sets": res, "pairs": pr}, open("/workspace/inv/results/baseline_test.json", "w"),
          indent=1)
print("\nBASELINE_DONE", flush=True)
