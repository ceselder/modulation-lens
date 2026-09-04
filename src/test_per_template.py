#!/usr/bin/env python3
"""
Is ONE template responsible for the wrapper bonus?

The grid has only 6 templates (10 of the original 16 died with the box), so a single outlier is 17%
of the signal and could carry the whole +0.079. Score pizza vs bare content per template, and per
carrier, instead of averaging them away.
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
NT, NC = len(C.TEMPLATES_RECOVERED), 3
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[:NC], 42, J, dev)
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
ALL = sorted(set(CORE.values()) | set(FULL.values()))


@torch.no_grad()
def read_one(strings, ti, ci, max_tok=32, batch=32):
    """read through ONE (template, carrier) cell only."""
    S = GRID.cells[ci][ti]
    out = {}
    ids_of = {s: (tok(s, add_special_tokens=False).input_ids[:max_tok]
                  or tok(" the", add_special_tokens=False).input_ids) for s in strings}
    buckets = {}
    for s, t in ids_of.items():
        buckets.setdefault(len(t), []).append(s)
    pre = torch.tensor(S["pre"], device=dev)
    post = torch.tensor(S["post"], device=dev)
    for _, grp in buckets.items():
        for a in range(0, len(grp), batch):
            ch = grp[a:a + batch]
            mid = torch.tensor([ids_of[s] for s in ch], device=dev)
            B = mid.shape[0]
            model(input_ids=torch.cat([pre.unsqueeze(0).expand(B, -1), mid,
                                       post.unsqueeze(0).expand(B, -1)], dim=1))
            v = L42["h"].float()[:, -S["ncar"]:, :].mean(1) @ J.T
            for kk, s in enumerate(ch):
                out[s] = v[kk]
    return out


# per-cell PMU, so each cell is centred by its OWN filler mean
import random
rr = random.Random(0)
words = ("policy river garden engine harbour lantern meadow cipher tunnel orchard beacon quarry "
         "saddle thistle vellum wharf pigment rafter cistern bramble").split()
fill = [" ".join(rr.choice(words) for _ in range(rr.randint(3, 9))) for _ in range(48)]

print("\n=== wrapper bonus per TEMPLATE (averaged over %d carriers) ===" % NC)
print("%-6s %10s %10s %10s   template" % ("tpl", "bare", "pizza", "bonus"))
res = {}
for ti in range(NT):
    bo, pz = [], []
    for ci in range(NC):
        fv = read_one(fill, ti, ci)
        pmu = torch.stack([fv[f] for f in fill]).mean(0)
        rv = read_one(ALL, ti, ci)
        for k in keys:
            for lst, s in ((bo, CORE[k]), (pz, FULL[k])):
                v = rv[s] - pmu
                lst.append(float((v / v.norm().clamp(min=1e-8)) @ TGT[k]))
    b, p = float(np.mean(bo)), float(np.mean(pz))
    res["tpl%d" % ti] = {"bare": b, "pizza": p, "bonus": p - b}
    print("%-6d %10.4f %10.4f %10.4f   %r"
          % (ti, b, p, p - b, C.TEMPLATES_RECOVERED[ti][:46]), flush=True)
bon = [res["tpl%d" % t]["bonus"] for t in range(NT)]
print("\n  bonus across templates: mean %+.4f  sd %.4f  min %+.4f (tpl%d)  max %+.4f (tpl%d)"
      % (np.mean(bon), np.std(bon), min(bon), int(np.argmin(bon)), max(bon), int(np.argmax(bon))))
print("  -> %s" % ("ONE TEMPLATE dominates; this is a template artifact"
                   if max(bon) > 2.5 * np.median(bon) else
                   "the bonus is broad across templates, not one bad template"))
json.dump(res, open("/workspace/inv/results/per_template.json", "w"), indent=1)
print("\nPER_TEMPLATE_DONE", flush=True)
