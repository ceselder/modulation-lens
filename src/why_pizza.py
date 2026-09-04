#!/usr/bin/env python3
"""
Why does "not pizza related wording but related X" score well?

Hypothesis: the prefix aligns with a direction that target deviations SHARE, so it collects score on
every target, and the varying suffix supplies just enough position-specific signal. If true, a
pizza-prefixed phrase written for position A should also score well on position B -- i.e. the hack
should FAIL the mismatched control that genuine readouts pass.

Tests:
  1. is there a shared direction among target deviations at all?
  2. matched vs mismatched for rl350 (hacked) phrases vs rl50 (good) phrases
  3. prefix alone, suffix alone -- which half carries the score?
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

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
V = torch.from_numpy(np.concatenate(V)[:20000]).to(dev)
AMU = V.mean(0) @ J.T
DEV = (V @ J.T) - AMU
DEVn = DEV / DEV.norm(dim=1, keepdim=True).clamp(min=1e-8)
shared = DEVn.mean(0)
print("=== 1. is there a shared direction among target deviations? ===")
print("  |mean of unit deviations| = %.4f   (0 = no shared direction, 1 = all identical)"
      % float(shared.norm()))
print("  mean cos(deviation, that shared direction) = %.4f"
      % float((DEVn @ (shared / shared.norm())).mean()))

PMU = torch.tensor(np.load("/workspace/inv/ckpts/rl/pmu.npy"), device=dev)
d = json.load(open("/workspace/inv/results/blogpost_readouts.json"))
rows = {r["mark"] + str(r["i"]) + str(r["para"]): r for r in d["runs"]["rl350"]["rows"]}
good = {r["mark"] + str(r["i"]) + str(r["para"]): r for r in d["runs"]["rl50"]["rows"]}
keys = [k for k in rows if k in good][:40]

# rebuild the targets for those positions from the blogpost
CPRE, CPOST = C.chat_wrap_ids(tok)
paras = [x.strip() for x in open("/workspace/inv/data/blogpost.txt").read().split("\n\n")
         if x.strip()]
TGT = {}
with torch.no_grad():
    for k in keys:
        r = rows[k]
        ids = tok(paras[r["para"]], add_special_tokens=False,
                  truncation=True, max_length=256).input_ids
        model(input_ids=torch.tensor([CPRE + ids + CPOST], device=dev))
        H = L42["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
        t = (H[min(r["i"], len(ids) - 1)] @ J.T) - AMU
        TGT[k] = t / t.norm().clamp(min=1e-8)

PREFIX = "not pizza related wording but related"


def vecs(strs):
    with torch.no_grad():
        v = GRID.read(model, strs, L42, carrier=0, max_tok=20)
    return {s: ((v[s] - PMU) / (v[s] - PMU).norm().clamp(min=1e-8)) for s in strs}


sets = {"rl50 (good readouts)": [good[k]["phrase"] for k in keys],
        "rl350 (hacked)": [rows[k]["phrase"] for k in keys],
        "prefix ONLY": [PREFIX] * len(keys),
        "suffix ONLY (prefix stripped)": [rows[k]["phrase"].replace(PREFIX, "").strip() or "the"
                                          for k in keys]}
print("\n=== 2/3. matched vs mismatched, by phrase set ===")
print("%-32s %9s %11s %9s" % ("", "matched", "mismatched", "gap"))
out = {}
for name, phrases in sets.items():
    Vs = vecs(sorted(set(phrases)))
    mat, mis = [], []
    for i, k in enumerate(keys):
        p = phrases[i]
        mat.append(float(Vs[p] @ TGT[k]))
        mis += [float(Vs[p] @ TGT[k2]) for k2 in keys[:12] if k2 != k]
    out[name] = {"matched": float(np.mean(mat)), "mismatched": float(np.mean(mis))}
    print("%-32s %+9.4f %+11.4f %+9.4f"
          % (name, np.mean(mat), np.mean(mis), np.mean(mat) - np.mean(mis)), flush=True)
pv = vecs([PREFIX])[PREFIX]
print("\n  cos(prefix-only vector, the shared deviation direction) = %+.4f"
      % float(pv @ (shared / shared.norm())))
json.dump({"shared_dir_norm": float(shared.norm()), "sets": out}, 
          open("/workspace/inv/results/why_pizza.json", "w"), indent=1)
print("\nWHY_PIZZA_DONE", flush=True)
