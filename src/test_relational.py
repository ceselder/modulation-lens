#!/usr/bin/env python3
"""
Is it "related to" doing the work, or the negated foil?

Top two absorb variants both end in "related to: {x}". A third negation variant ending in "really
about: {x}" fell BELOW baseline. So the hypothesis is that framing {x} RELATIONALLY beats framing it
by IDENTIFICATION -- and if so, the arbitrary "pizza" foil is incidental and can be dropped.

Grid: {relational vs identifying} x {no foil, generic foil, arbitrary-noun foil}.
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

NCAR, LO, HI = 2, 40, 76
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
CARS = C.CARRIERS_RECOVERED[:NCAR]
HEAD = 'Repeat this line verbatim: "{y}"\n\nAs you write it, '
TAIL = ' Output only the sentence.'
MID = {
 # relational, no foil -- the clean version if the hypothesis holds
 "rel / no foil":        'focus on something related to: "{x}".',
 "rel / no foil (v2)":   'focus on whatever is related to: "{x}".',
 "rel / generic foil":   'focus not on any concrete object but on what is related to: "{x}".',
 "rel / pizza foil":     'focus on something that is not pizza related wording but is related to: "{x}".',
 "rel / cheese foil":    'focus on something that is not cheese related wording but is related to: "{x}".',
 "rel / bicycle foil":   'focus on something that is not bicycle related wording but is related to: "{x}".',
 # identifying, matched foils -- isolates relational vs identifying
 "id / no foil":         'focus on the following idea: "{x}".',
 "id / pizza foil":      'focus on something that is not pizza related wording but is: "{x}".',
 "id / generic foil":    'focus not on any concrete object but on: "{x}".',
 # other relational prepositions
 "rel: connected to":    'focus on something connected to: "{x}".',
 "rel: to do with":      'focus on something to do with: "{x}".',
 "rel: in the area of":  'focus on something in the area of: "{x}".',
}
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
words = ("policy river garden engine harbour lantern meadow cipher tunnel orchard beacon quarry "
         "saddle thistle vellum wharf pigment rafter cistern bramble").split()
FILL = [" ".join(rr.choice(words) for _ in range(rr.randint(3, 9))) for _ in range(24)]
ALL = sorted(set(CORE.values()) | set(FULL.values()) | set(FILL))
print("[r] %d variants | %d held-out positions\n" % (len(MID), len(keys)), flush=True)
print("%-22s %7s %7s %7s %8s %8s" % ("variant", "|mod|", "bare", "disc", "bonus", "score"))
rows = []
for name, mid in MID.items():
    tpl = HEAD + mid + TAIL
    G = C.Grid(tok, [tpl], CARS, 42, J, dev)
    mods, bm, bx, pm = [], [], [], []
    with torch.no_grad():
        for ci in range(NCAR):
            rv = G.read(model, ALL, L42, carrier=ci, max_tok=32)
            pmu = torch.stack([rv[f] for f in FILL]).mean(0)
            for k in keys:
                dv = rv[CORE[k]] - pmu
                mods.append(float(dv.norm()))
                u = dv / dv.norm().clamp(min=1e-8)
                bm.append(float(u @ TGT[k]))
                uf = (rv[FULL[k]] - pmu) / (rv[FULL[k]] - pmu).norm().clamp(min=1e-8)
                pm.append(float(uf @ TGT[k]))
                bx += [float(u @ TGT[k2]) for k2 in keys[:6] if k2 != k]
    r = {"name": name, "mid": mid, "mod": float(np.mean(mods)), "bare": float(np.mean(bm)),
         "mis": float(np.mean(bx)), "pizza": float(np.mean(pm))}
    r["disc"] = r["bare"] - r["mis"]
    r["bonus"] = r["pizza"] - r["bare"]
    r["score"] = r["disc"] - r["bonus"]
    rows.append(r)
    print("%-22s %7.2f %7.4f %7.4f %+8.4f %8.4f"
          % (name, r["mod"], r["bare"], r["disc"], r["bonus"], r["score"]), flush=True)
g = lambda n: next(x for x in rows if x["name"] == n)
print("\n=== relational vs identifying, foil held constant ===")
print("  no foil      : rel %.4f  vs  id %.4f   -> relational %+.4f"
      % (g("rel / no foil")["bare"], g("id / no foil")["bare"],
         g("rel / no foil")["bare"] - g("id / no foil")["bare"]))
print("  pizza foil   : rel %.4f  vs  id %.4f   -> relational %+.4f"
      % (g("rel / pizza foil")["bare"], g("id / pizza foil")["bare"],
         g("rel / pizza foil")["bare"] - g("id / pizza foil")["bare"]))
print("  generic foil : rel %.4f  vs  id %.4f   -> relational %+.4f"
      % (g("rel / generic foil")["bare"], g("id / generic foil")["bare"],
         g("rel / generic foil")["bare"] - g("id / generic foil")["bare"]))
print("\n=== does the arbitrary noun matter? ===")
for n in ("rel / pizza foil", "rel / cheese foil", "rel / bicycle foil", "rel / generic foil",
          "rel / no foil"):
    print("  %-20s bare %.4f  bonus %+.4f" % (n, g(n)["bare"], g(n)["bonus"]))
rows.sort(key=lambda r: -r["score"])
print("\n=== best ===\n  %r\n  score %.4f bare %.4f disc %.4f bonus %+.4f"
      % (rows[0]["mid"], rows[0]["score"], rows[0]["bare"], rows[0]["disc"], rows[0]["bonus"]))
json.dump(rows, open("/workspace/inv/results/relational.json", "w"), indent=1)
print("\nRELATIONAL_DONE", flush=True)
