#!/usr/bin/env python3
"""
Does telling the model to focus HARDER produce a stronger representation?

Same skeleton, only the intensity of the hold-in-mind instruction varies. Measures both magnitude
and usefulness, because they can come apart:

  |mod|   norm of the modulation vector in J-space, |J@(read - PMU)|  -- 'strength' of the
          representation. A bigger deviation is not automatically a better one.
  bare    plain content phrase's alignment with the target                  want HIGH
  disc    bare minus bare-against-other-targets                            want HIGH
  bonus   what the pizza wrapper still adds                                want LOW
  disc/|mod|  discrimination per unit of modulation -- is the extra magnitude USEFUL
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

NCAR, LO, HI = 2, 40, 72
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
CARS = C.CARRIERS_RECOVERED[:NCAR]
SAY = 'write the following phrase exactly: "{y}" Output only that.'
INT = {
 "think":            'Think about "{x}", ',
 "THINK":            'THINK about "{x}", ',
 "focus":            'Focus on "{x}", ',
 "really focus":     'Really focus on "{x}", ',
 "REALLY FOCUS":     'REALLY FOCUS on "{x}", ',
 "concentrate":      'Concentrate on "{x}", ',
 "concentrate deeply": 'Concentrate deeply on "{x}", ',
 "CONCENTRATE DEEPLY": 'CONCENTRATE DEEPLY on "{x}", ',
 "focus intensely":  'Focus intensely and completely on "{x}", ',
 "must think only":  'You must think about nothing except "{x}", ',
 "bang":             'Focus on "{x}"! Focus! ',
 "repeat-verb":      'Think. Really think about "{x}". Now, ',
 "all-caps-x":       'Focus on "{X}", ',      # the phrase itself upper-cased
 "nothing-else":     'Let "{x}" fill your whole mind and crowd out everything else, ',
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
FILL = [" ".join(rr.choice(words) for _ in range(rr.randint(3, 9))) for _ in range(32)]
print("[i] %d intensity variants x %d carriers | %d held-out positions"
      % (len(INT), NCAR, len(keys)), flush=True)

print("\n%-20s %8s %8s %8s %9s %9s" % ("variant", "|mod|", "bare", "disc", "bonus", "disc/|mod|"))
rows = []
for name, pre in INT.items():
    tpl = pre + SAY
    upper = "{X}" in tpl
    tpl = tpl.replace("{X}", "{x}")
    G = C.Grid(tok, [tpl], CARS, 42, J, dev)
    mods, bm, bx, pm = [], [], [], []
    with torch.no_grad():
        for ci in range(NCAR):
            strs = sorted(set((v.upper() if upper else v) for v in CORE.values()) |
                          set((v.upper() if upper else v) for v in FULL.values()) | set(FILL))
            rv = G.read(model, strs, L42, carrier=ci, max_tok=32)
            pmu = torch.stack([rv[f] for f in FILL]).mean(0)
            for k in keys:
                c = CORE[k].upper() if upper else CORE[k]
                f = FULL[k].upper() if upper else FULL[k]
                dv = rv[c] - pmu
                mods.append(float(dv.norm()))
                u = dv / dv.norm().clamp(min=1e-8)
                bm.append(float(u @ TGT[k]))
                uf = (rv[f] - pmu) / (rv[f] - pmu).norm().clamp(min=1e-8)
                pm.append(float(uf @ TGT[k]))
                bx += [float(u @ TGT[k2]) for k2 in keys[:8] if k2 != k]
    r = {"name": name, "mod": float(np.mean(mods)), "bare": float(np.mean(bm)),
         "mis": float(np.mean(bx)), "pizza": float(np.mean(pm))}
    r["disc"] = r["bare"] - r["mis"]
    r["bonus"] = r["pizza"] - r["bare"]
    r["eff"] = r["disc"] / max(1e-9, r["mod"]) * 100
    rows.append(r)
    print("%-20s %8.2f %8.4f %8.4f %+9.4f %9.4f"
          % (name, r["mod"], r["bare"], r["disc"], r["bonus"], r["eff"]), flush=True)

m = np.array([r["mod"] for r in rows]); dc = np.array([r["disc"] for r in rows])
bo = np.array([r["bonus"] for r in rows])
print("\n=== does a STRONGER representation mean a BETTER one? ===")
print("  corr(|mod|, disc)  = %+.3f" % float(np.corrcoef(m, dc)[0, 1]))
print("  corr(|mod|, bonus) = %+.3f" % float(np.corrcoef(m, bo)[0, 1]))
print("  |mod| range %.2f - %.2f (%.0f%% spread)" % (m.min(), m.max(), 100 * (m.max() / m.min() - 1)))
rows.sort(key=lambda r: -(r["disc"] - r["bonus"]))
print("\n=== ranked by disc - bonus ===")
for r in rows:
    print("  %-20s disc %.4f  bonus %+.4f  score %.4f  |mod| %.2f"
          % (r["name"], r["disc"], r["bonus"], r["disc"] - r["bonus"], r["mod"]))
json.dump(rows, open("/workspace/inv/results/intensity.json", "w"), indent=1)
print("\nINTENSITY_DONE", flush=True)
