#!/usr/bin/env python3
"""
Four ways to compare a modulation vector to a target activation.

  modulation vector m(p) = mean-pool over the forced carrier positions in the assistant turn,
                           minus the mean over many other slot fillers (the batch mean)

axes:
  jacobian  : J @ m(p)  vs  J @ h        -- or raw m(p) vs h, no J at all
  target    : centred (h - AMU) or raw h

Scored by matched-vs-mismatched separation: does a position's own label beat labels harvested at
OTHER positions? That is the only thing that makes a comparison well-posed, and it is invariant to
whichever variant we pick, so the four are directly comparable. Unwhitened cosine throughout so
whitening is not a confound (the existing whitener is J-space only; a raw-L42 one would have to be
refitted if no-J wins).
"""
import argparse, glob, json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/inv/data/prose_L42.parquet")
ap.add_argument("--n", type=int, default=160)
ap.add_argument("--n-mis", type=int, default=24)
ap.add_argument("--carriers", type=int, default=3)
ap.add_argument("--layer", type=int, default=42)
ap.add_argument("--min-words", type=int, default=4)
ap.add_argument("--out", default="/workspace/inv/results/ablate_jacobian.json")
A = ap.parse_args()
dev = "cuda"
rng = np.random.default_rng(0)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[A.layer].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(A.layer, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[: A.carriers], A.layer, J, dev)

# ---- reads WITHOUT J, so both variants come from one pass over the grid ----
@torch.no_grad()
def read_raw(strings, carrier=0, max_tok=24, batch=40):
    ids_of = {s: (tok(s, add_special_tokens=False).input_ids[:max_tok]
                  or tok(" the", add_special_tokens=False).input_ids) for s in strings}
    acc = {s: torch.zeros(C.D_MODEL, device=dev) for s in strings}
    buckets = {}
    for s, t in ids_of.items():
        buckets.setdefault(len(t), []).append(s)
    for S in GRID.cells[carrier % GRID.n_car]:
        pre = torch.tensor(S["pre"], device=dev)
        post = torch.tensor(S["post"], device=dev)
        for _, grp in buckets.items():
            for a in range(0, len(grp), batch):
                ch = grp[a:a + batch]
                mid = torch.tensor([ids_of[s] for s in ch], device=dev)
                B = mid.shape[0]
                model(input_ids=torch.cat([pre.unsqueeze(0).expand(B, -1), mid,
                                           post.unsqueeze(0).expand(B, -1)], dim=1))
                v = L42["h"].float()[:, -S["ncar"]:, :].mean(1)      # RAW L42, no J
                for k, s in enumerate(ch):
                    acc[s] += v[k]
    return {s: acc[s] / GRID.n_tpl for s in strings}


V, L = [], []
for b in pq.ParquetFile(A.data).iter_batches(batch_size=4096,
                                             columns=["activation_vector", "label"]):
    d = b.to_pydict()
    V.append(np.array(d["activation_vector"], dtype="float32"))
    L.extend(d["label"])
    if sum(len(x) for x in V) >= 20000:
        break
V = np.concatenate(V)
keep = [i for i, l in enumerate(L) if len(l.split()) >= A.min_words]
AMU_raw = torch.from_numpy(V[keep]).mean(0).to(dev)                  # raw L42 pool mean
sel = [keep[i] for i in rng.choice(len(keep), size=A.n, replace=False)]
print("[ab] %d positions | grid %dx%d" % (len(sel), GRID.n_tpl, A.carriers), flush=True)

# batch mean over slot fillers, raw L42 -- this is the "minus the mean over the batch" term
import random
rr = random.Random(0)
words = ("policy river garden engine harbour lantern meadow cipher tunnel orchard beacon quarry "
         "saddle thistle vellum wharf pigment rafter cistern bramble").split()
fill = [" ".join(rr.choice(words) for _ in range(rr.randint(3, 9))) for _ in range(64)]
PMU_raw = torch.stack(list(read_raw(fill).values())).mean(0)
print("[ab] |PMU_raw| %.2f  |AMU_raw| %.2f  |J@PMU| %.2f  |J@AMU| %.2f"
      % (float(PMU_raw.norm()), float(AMU_raw.norm()),
         float((PMU_raw @ J.T).norm()), float((AMU_raw @ J.T).norm())), flush=True)

labs = sorted({L[i] for i in sel})
print("[ab] reading %d labels through the grid..." % len(labs), flush=True)
RAW = read_raw(labs)
MOD_raw = {s: RAW[s] - PMU_raw for s in labs}          # the modulation vector, raw L42
MOD_j = {s: (RAW[s] - PMU_raw) @ J.T for s in labs}    # J applied to the modulation vector


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


VARIANTS = {
    "J,  target centred   (current)": lambda h: ((h - AMU_raw) @ J.T, MOD_j),
    "J,  target raw":                 lambda h: (h @ J.T,             MOD_j),
    "noJ, target centred":            lambda h: (h - AMU_raw,         MOD_raw),
    "noJ, target raw":                lambda h: (h,                   MOD_raw),
}
res = {}
for name, f in VARIANTS.items():
    own, mis, wins = [], [], []
    for i in sel:
        h = torch.from_numpy(V[i]).to(dev)
        t, MOD = f(h)
        o = cos(MOD[L[i]], t)
        others = [j for j in sel if L[j] != L[i]]
        pick = rng.choice(len(others), size=min(A.n_mis, len(others)), replace=False)
        m = [cos(MOD[L[others[k]]], t) for k in pick]
        own.append(o); mis.append(float(np.mean(m))); wins.append(o > max(m))
    own, mis, wins = np.array(own), np.array(mis), np.array(wins)
    d = (own.mean() - mis.mean()) / max(1e-9, own.std())
    res[name] = {"matched": float(own.mean()), "mismatched": float(mis.mean()),
                 "sep": float(own.mean() - mis.mean()), "cohens_d": float(d),
                 "win_rate": float(wins.mean())}
    print("  %-32s matched %+.4f  mismatched %+.4f  sep %+.4f  d %.2f  win %.0f%%"
          % (name, own.mean(), mis.mean(), own.mean() - mis.mean(), d, 100 * wins.mean()),
          flush=True)
json.dump(res, open(A.out, "w"), indent=1)
print("\nchance win rate = %.0f%%" % (100 / (A.n_mis + 1)))
print("ABLATE_DONE", flush=True)
