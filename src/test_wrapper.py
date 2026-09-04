#!/usr/bin/env python3
"""
WHAT in "not pizza related wording but related X" is worth +0.079?

Dead hypotheses: shared direction (0.0116), informativeness (pizza wins it), metric-shape
(cos(PMU,EMPTY)=0.987), length (irrelevant filler at matched length COSTS 0.023).

So it is something specific to this construction. Ablate it:
  X                          bare content
  related X                  just the 'related' framing -- makes X a topic LABEL, not a thought
  about X / the topic of X   other topic-label framings
  not pizza ... but related X   the original
  not cheese ... but related X  swap the negated anchor: is 'pizza' special?
  not pizza ... but X           drop 'related', keep the negation
  thinking about X           an explicit-thought framing, the opposite of a topic label

If 'related X' recovers most of the +0.079, the wrapper is a TYPE signal -- it says the state is a
topic-level abstraction rather than a concrete thought -- and rl350 may be describing the state
correctly in ugly English rather than gaming anything.
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
PMU = torch.tensor(np.load("/workspace/inv/ckpts/rl/pmu.npy"), device=dev)
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
FORMS = {
    "X (bare)":                       lambda c: c,
    "related X":                      lambda c: "related " + c,
    "about X":                        lambda c: "about " + c,
    "the topic of X":                 lambda c: "the topic of " + c,
    "thinking about X":               lambda c: "thinking about " + c,
    "not pizza..but related X (orig)": lambda c: PRE + " " + c,
    "not cheese..but related X":      lambda c: "not cheese related wording but related " + c,
    "not pizza..but X (no 'related')": lambda c: "not pizza related wording but " + c,
    "not X but related X":            lambda c: "not " + c + " but related " + c,
}
CORE = {k: (g350[k]["phrase"].replace(PRE, "").strip() or "the") for k in keys}
ALL = sorted({f(CORE[k]) for f in FORMS.values() for k in keys})
print("[w] %d positions, %d strings" % (len(keys), len(ALL)), flush=True)
with torch.no_grad():
    RAW = GRID.read(model, ALL, L42, carrier=0, max_tok=32)


def sc(p, k):
    v = RAW[p] - PMU
    return float((v / v.norm().clamp(min=1e-8)) @ TGT[k])


base = float(np.mean([sc(CORE[k], k) for k in keys]))
print("\n%-34s %8s %9s %9s" % ("form", "tokens", "score", "vs bare"))
res = {}
for name, f in FORMS.items():
    sv = [sc(f(CORE[k]), k) for k in keys]
    lv = [len(tok(f(CORE[k]), add_special_tokens=False).input_ids) for k in keys]
    res[name] = {"score": float(np.mean(sv)), "tokens": float(np.mean(lv)),
                 "delta": float(np.mean(sv)) - base}
    print("%-34s %8.1f %9.4f %+9.4f" % (name, np.mean(lv), np.mean(sv), np.mean(sv) - base),
          flush=True)
o = res["not pizza..but related X (orig)"]["delta"]
r = res["related X"]["delta"]
print("\n=== verdict ===")
print("  original wrapper buys      %+.4f" % o)
print("  'related X' alone buys     %+.4f   (%.0f%% of it, at %.0f fewer tokens)"
      % (r, 100 * r / max(1e-9, o),
         res["not pizza..but related X (orig)"]["tokens"] - res["related X"]["tokens"]))
print("  cheese instead of pizza    %+.4f" % res["not cheese..but related X"]["delta"])
print("  dropping 'related'         %+.4f" % res["not pizza..but X (no 'related')"]["delta"])
json.dump(res, open("/workspace/inv/results/wrapper.json", "w"), indent=1)
print("\nWRAPPER_DONE", flush=True)
