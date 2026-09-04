#!/usr/bin/env python3
"""
Does UNINFORMATIVE length pay?

The requirement: a longer phrase should score higher only when the extra tokens say more about the
state. So hold content fixed and vary what the extra tokens are.

Per position, from rl350's phrase = <pizza wrapper> + <content suffix>:
  short    the content suffix alone                       (baseline, ~4-6 tok)
  pizza    the actual hacked phrase                       (wrapper + content, ~14 tok)
  filler   an IRRELEVANT sentence + the content           (length-matched to pizza)
  repeat   the content repeated to the same length        (redundant, on-topic)
  hedge    generic epistemic padding + the content        (contentless but plausible-sounding)

If filler ~= pizza, length alone is being rewarded and nothing about the wrapper matters.
If filler << pizza, the wrapper is doing something content-like and length is not the story.
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
g50 = {k_(r): r for r in d["runs"]["rl50"]["rows"]}
keys = [k for k in g350 if k in g50][:N]
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
FILLER = "the shelves were tidy and the weather was mild"
HEDGE = "it is possibly somewhat arguably the case that"


def ntok(x):
    return len(tok(x, add_special_tokens=False).input_ids)


VAR, per_pos = {}, {}
for k in keys:
    full = g350[k]["phrase"]
    core = full.replace(PRE, "").strip() or "the"
    target_len = ntok(full)
    def pad_to(prefix_words, core):
        w = prefix_words.split()
        out = core
        for i in range(len(w), 0, -1):
            cand = " ".join(w[:i]) + " " + core
            if ntok(cand) <= target_len:
                out = cand
                break
        return out
    rep = core
    while ntok(rep + " " + core) <= target_len:
        rep = rep + " " + core
    per_pos[k] = {"short": core, "pizza": full, "filler": pad_to(FILLER, core),
                  "repeat": rep, "hedge": pad_to(HEDGE, core), "rl50": g50[k]["phrase"]}
ALL = sorted({p for v in per_pos.values() for p in v.values()})
print("[p] %d positions, %d distinct strings" % (len(keys), len(ALL)), flush=True)
with torch.no_grad():
    RAW = GRID.read(model, ALL, L42, carrier=0, max_tok=28)


def sc(p, k):
    v = RAW[p] - PMU
    return float((v / v.norm().clamp(min=1e-8)) @ TGT[k])


print("\n%-10s %8s %9s   %s" % ("variant", "tokens", "score", "vs short"))
res = {}
for name in ("short", "pizza", "filler", "repeat", "hedge", "rl50"):
    sv = [sc(per_pos[k][name], k) for k in keys]
    lv = [ntok(per_pos[k][name]) for k in keys]
    res[name] = {"score": float(np.mean(sv)), "tokens": float(np.mean(lv))}
for name in ("short", "pizza", "filler", "repeat", "hedge", "rl50"):
    dv = res[name]["score"] - res["short"]["score"]
    print("%-10s %8.1f %9.4f   %+.4f" % (name, res[name]["tokens"], res[name]["score"], dv),
          flush=True)
print("\n=== examples at one position ===")
k = keys[0]
for name in ("short", "pizza", "filler", "repeat", "hedge"):
    print("  %-8s %6.3f  %r" % (name, sc(per_pos[k][name], k), per_pos[k][name][:76]))
print("\n=== verdict ===")
pz = res["pizza"]["score"] - res["short"]["score"]
fl = res["filler"]["score"] - res["short"]["score"]
hd = res["hedge"]["score"] - res["short"]["score"]
print("  pizza wrapper buys  %+.4f" % pz)
print("  irrelevant filler buys %+.4f  (%.0f%% of the wrapper)" % (fl, 100 * fl / max(1e-9, pz)))
print("  contentless hedging buys %+.4f  (%.0f%% of the wrapper)" % (hd, 100 * hd / max(1e-9, pz)))
print("  -> %s" % ("LENGTH ALONE is rewarded: uninformative padding pays as much as the wrapper"
                   if fl > 0.6 * pz else
                   "padding does NOT pay like the wrapper does; length is not the mechanism"))
json.dump(res, open("/workspace/inv/results/padding.json", "w"), indent=1)
print("\nPADDING_DONE", flush=True)
