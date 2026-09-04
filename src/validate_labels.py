#!/usr/bin/env python3
"""
Decisive go/no-go on the preceding-clause label, at scale and in the metric the reward uses.

The first pass was n=12, unwhitened, uniformly-sampled positions: 14x matched-vs-mismatched
separation but the own-label won outright at only 7/12, and every failure ended mid-clause. The
harvest now samples clause boundaries, so this re-tests with whitening on and enough n to also ask
whether label LENGTH predicts label quality -- the harvest still contains things like 'D,' and
'for two months,' that carry almost no content.
"""
import argparse, collections, json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/inv/data/prose_L42.parquet")
ap.add_argument("--whitener", default="/workspace/inv/data/meansub/natural_whitener_jspace.npz")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--n-mis", type=int, default=24, help="wrong labels compared per position")
ap.add_argument("--layer", type=int, default=42)
ap.add_argument("--ridge", default="0.1")
ap.add_argument("--carriers", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
dev = "cuda"
rng = np.random.default_rng(A.seed)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[A.layer].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(A.layer, dev)
MU, Wm = C.load_whitener(A.whitener, A.ridge, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[: A.carriers], A.layer, J, dev)
print("[v] grid %d tpl x %d car | whitener |mu| %.2f"
      % (GRID.n_tpl, GRID.n_car, float(MU.norm())), flush=True)

pf = pq.ParquetFile(A.data)
V, L = [], []
for b in pf.iter_batches(batch_size=4096, columns=["activation_vector", "label"]):
    d = b.to_pydict()
    V.append(np.array(d["activation_vector"], dtype="float32"))
    L.extend(d["label"])
    if sum(len(x) for x in V) >= 20000:
        break
V = np.concatenate(V)
AMU = torch.from_numpy(V).mean(0).to(dev) @ J.T
sel = rng.choice(len(L), size=A.n, replace=False)
print("[v] pool %d | testing %d positions" % (len(L), A.n), flush=True)

PMU = GRID.prompt_mean(model, L42, n=64)
print("[v] |PMU| %.2f" % float(PMU.norm()), flush=True)

labs = sorted({L[i] for i in sel})
print("[v] reading %d distinct labels through the grid..." % len(labs), flush=True)
vec = GRID.read(model, labs, L42, carrier=0, max_tok=24)
CAND = {s: ((vec[s] - PMU) @ Wm.T) for s in labs}
for s in CAND:
    CAND[s] = CAND[s] / CAND[s].norm().clamp(min=1e-8)

rows = []
for i in sel:
    t = ((torch.from_numpy(V[i]).to(dev) @ J.T) - AMU) @ Wm.T
    t = t / t.norm().clamp(min=1e-8)
    own = float(CAND[L[i]] @ t)
    others = [j for j in sel if L[j] != L[i]]
    pick = rng.choice(len(others), size=min(A.n_mis, len(others)), replace=False)
    mis = [float(CAND[L[others[k]]] @ t) for k in pick]
    rows.append({"label": L[i], "words": len(L[i].split()), "own": own,
                 "mis_mean": float(np.mean(mis)), "mis_max": float(np.max(mis)),
                 "wins": own > max(mis)})

own = np.array([r["own"] for r in rows])
mism = np.array([r["mis_mean"] for r in rows])
wins = np.array([r["wins"] for r in rows])
print("\n=== whitened, boundary labels, n=%d ===" % len(rows))
print("matched      mean %.4f  sd %.4f" % (own.mean(), own.std()))
print("mismatched   mean %.4f  sd %.4f" % (mism.mean(), mism.std()))
print("separation   %+.4f" % (own.mean() - mism.mean()))
print("own label beats all %d wrong labels: %d/%d = %.0f%%  (chance %.0f%%)"
      % (A.n_mis, wins.sum(), len(wins), 100 * wins.mean(), 100 / (A.n_mis + 1)))
print("\n=== does label length predict quality? ===")
print("%-14s %5s %8s %8s" % ("words", "n", "matched", "win rate"))
for lo, hi in [(1, 3), (4, 6), (7, 9), (10, 13), (14, 99)]:
    m = np.array([lo <= r["words"] <= hi for r in rows])
    if m.sum() >= 5:
        print("%-14s %5d %8.4f %7.0f%%" % ("%d-%d" % (lo, hi), m.sum(),
              own[m].mean(), 100 * wins[m].mean()))
print("\n=== best and worst ===")
rows.sort(key=lambda r: -r["own"])
for r in rows[:5]:
    print("  %+.4f  %r" % (r["own"], r["label"][:72]))
print("  ...")
for r in rows[-5:]:
    print("  %+.4f  %r" % (r["own"], r["label"][:72]))
json.dump(rows, open("/workspace/inv/results/validate_labels.json", "w"), indent=1)
print("\nVALIDATE_DONE", flush=True)
