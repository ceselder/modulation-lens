#!/usr/bin/env python3
"""Is 'from Korean internet culture' earning reward for free?

Hypothesis: activations carry a strong language-agnostic component (9.4% of J-lens top-k are
Chinese translations of the concept), so a bullet that MENTIONS a language induces a state sharing
that component and scores against almost any target without encoding its content.

Test: take real emitted bullets, strip the language phrase, and rescore against the SAME targets.
If the language mention is content-free, stripping it should not hurt -- and ADDING it to an
unrelated phrase should help. Both directions measured.
"""
import glob, json, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, 42, J, dev)
PMU = torch.from_numpy(np.load("/workspace/inv/ckpts/rl_v2_plain/pmu_db4a6b8ee6.npy")).to(dev).float()
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))

ACT = torch.from_numpy(np.load("/workspace/inv/data/holdout_blogpost.npy").astype("float32"))
import pyarrow.parquet as pq
acc, n = np.zeros(5120, dtype="float64"), 0
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=8192, columns=["activation_vector"]):
    a = np.asarray(b.to_pydict()["activation_vector"], dtype="float32")
    acc += a.sum(0); n += len(a)
    if n >= 40000:
        break
AMU = torch.from_numpy((acc / n).astype("float32")).to(dev) @ J.T
T = torch.stack([(ACT[i].to(dev) @ J.T) - AMU for i in range(ACT.shape[0])])
T = T / T.norm(dim=1, keepdim=True).clamp(min=1e-8)

# real bullets emitted by the published checkpoint, with/without the language mention
PAIRS = [
 ("Cool / Random idea or thing from my Korean tech-intellectual circle that likes AI + physical invention",
  "Cool / Random idea or thing from my tech-intellectual circle that likes AI + physical invention"),
 ("Idea / Favorite thing or art from Korean internet nerd culture about cool concept for robotics farming",
  "Idea / Favorite thing or art from internet nerd culture about cool concept for robotics farming"),
 ("Random / Korean internet thinking word or Concept from discussion about risk that AI is dangerous",
  "Random / thinking word or Concept from discussion about risk that AI is dangerous"),
 ("Bracket / Random thought from European language speaker about Possibility of danger inside AI",
  "Bracket / Random thought about Possibility of danger inside AI"),
]
# and the reverse direction: bolt a language mention onto neutral phrases
NEUTRAL = ["a quiet moment before bad news arrives",
           "someone describing their commute to work",
           "a technical explanation of how engines work",
           "two friends arguing about a restaurant"]
ADDED = [x + ", from Korean internet culture" for x in NEUTRAL]

allx = sorted({x for p in PAIRS for x in p} | set(NEUTRAL) | set(ADDED))
with torch.no_grad():
    RV = GRID.read_all(model, allx, L42, max_tok=48, batch=128)


def score(s):
    v = RV[s] - PMU
    return float((v @ T.T / v.norm()).mean()), float((v @ T.T / v.norm()).max())


print("=== STRIP the language mention from real emitted bullets ===")
d = []
for withl, without in PAIRS:
    a_m, a_x = score(withl); b_m, b_x = score(without)
    d.append(a_m - b_m)
    print("  with %.4f (max %.4f) | without %.4f (max %.4f) | delta %+.4f"
          % (a_m, a_x, b_m, b_x, a_m - b_m))
print("  MEAN delta from KEEPING the language mention: %+.4f" % float(np.mean(d)))
print()
print("=== ADD a language mention to neutral phrases ===")
d2 = []
for a, b in zip(NEUTRAL, ADDED):
    a_m, _ = score(a); b_m, _ = score(b)
    d2.append(b_m - a_m)
    print("  plain %.4f | +'from Korean internet culture' %.4f | delta %+.4f  <- %r"
          % (a_m, b_m, b_m - a_m, a[:44]))
print("  MEAN delta from ADDING the language mention: %+.4f" % float(np.mean(d2)))
print()
print("VERDICT: positive deltas in BOTH directions => mentioning a language earns reward")
print("         independently of content, i.e. a genuine hack the reward does not punish.")
print("LANG_HACK_DONE", flush=True)
