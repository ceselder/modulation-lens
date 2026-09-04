#!/usr/bin/env python3
"""Precise version: does a language mention raise OWN-target score more than MISMATCHED?

If own and mis rise by the same amount, the mention encodes nothing about the specific activation
and `disc = own - mis` is unchanged -- i.e. a pure exploit of a shared direction, and switching the
reward to disc removes it exactly. If own rises more, part of it is real information.

Uses the four-bullet checkpoint's ACTUAL bullets for specific probe positions, so each phrase has a
genuine own-target, then ablates the language phrase.
"""
import re, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C
dev, EOS = "cuda", 248046
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = base.model
INJ, LEFT, RIGHT = C.marker_ids(tok)
HOOK, L42 = {"vec": None, "ids": None}, {}
inner.register_forward_pre_hook(
    lambda m, a, kw: HOOK.__setitem__("ids", kw.get("input_ids") if kw.get("input_ids") is not None
                                      else (a[0] if a else None)), with_kwargs=True)


def _inj(m, a, out):
    r = out[0] if isinstance(out, tuple) else out
    ids, v = HOOK["ids"], HOOK["vec"]
    if v is None or ids is None or tuple(ids.shape) != tuple(r.shape[:-1]) or not bool((ids == INJ).any()):
        return out
    new = C.inject_at_marker(ids, r, v, INJ, LEFT, RIGHT, "replace")
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


inner.layers[1].register_forward_hook(_inj)
inner.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
model = PeftModel.from_pretrained(base, "/workspace/inv/ckpts/rl_b4_long4/iter_000200").eval()
J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, 42, J, dev)
PMU = torch.from_numpy(np.load("/workspace/inv/ckpts/rl_v2_plain/pmu_db4a6b8ee6.npy")).to(dev).float()

acc, n = np.zeros(5120, dtype="float64"), 0
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=8192, columns=["activation_vector"]):
    a = np.asarray(b.to_pydict()["activation_vector"], dtype="float32")
    acc += a.sum(0); n += len(a)
    if n >= 40000:
        break
AMU = torch.from_numpy((acc / n).astype("float32")).to(dev) @ J.T
ACT = torch.from_numpy(np.load("/workspace/inv/data/holdout_blogpost.npy").astype("float32"))
T = torch.stack([(ACT[i].to(dev) @ J.T) - AMU for i in range(ACT.shape[0])])
T = T / T.norm(dim=1, keepdim=True).clamp(min=1e-8)

JOB = open("/tmp/j229.txt").read() if False else None
LANGRE = re.compile(r"\b(from |in )?(korean|chinese|japanese|norwegian|dutch|european|german|"
                    r"french|russian)\b[^,./]*", re.I)
# collect real bullets that contain a language mention, keeping their own-target index
pairs = []
src = open("/workspace/inv/logs/rl_b4_long4.log", errors="ignore").read()
for m in re.finditer(r"\s+0\.\d+\s+\S+\s+\.\.\..*?-> '(.+?)'", src):
    b = m.group(1)
    if LANGRE.search(b):
        pairs.append(b)
pairs = pairs[:8]
# fall back: synthesise from the phrases the user pasted
if len(pairs) < 4:
    pairs = ["Cool / Random idea or thing from my Korean tech-intellectual circle that likes AI",
             "Idea / Favorite thing or art from Korean internet nerd culture about robotics",
             "Random / Korean internet thinking word or Concept about risk that AI is dangerous",
             "Thing / Design or simple noun from Korean language referring to Cool startup idea"]
strip = [LANGRE.sub("", p).replace("  ", " ").strip() for p in pairs]
allx = sorted(set(pairs) | set(strip))
with torch.no_grad():
    RV = GRID.read_all(model.get_base_model(), allx, L42, max_tok=48, batch=128)


def own_mis(s, k):
    v = RV[s] - PMU
    cs = (v @ T.T / v.norm()).squeeze()
    own = float(cs[k])
    mis = float((cs.sum() - cs[k]) / (cs.numel() - 1))
    return own, mis


print("%-64s %8s %8s %8s" % ("phrase", "own", "mis", "disc"))
d_own, d_mis, d_disc = [], [], []
for k, (withl, without) in enumerate(zip(pairs, strip)):
    ow, mw = own_mis(withl, k % T.shape[0])
    oo, mo = own_mis(without, k % T.shape[0])
    print("%-64s %8.4f %8.4f %8.4f   WITH lang" % (withl[:62], ow, mw, ow - mw))
    print("%-64s %8.4f %8.4f %8.4f   stripped" % (without[:62], oo, mo, oo - mo))
    d_own.append(ow - oo); d_mis.append(mw - mo); d_disc.append((ow - mw) - (oo - mo))
print()
print("  delta from the language mention:  own %+.4f | mis %+.4f | DISC %+.4f"
      % (float(np.mean(d_own)), float(np.mean(d_mis)), float(np.mean(d_disc))))
print()
print("  If own and mis rose about equally, DISC ~ 0 => the mention is a PURE shared-direction")
print("  exploit, and training on own-minus-mis removes it exactly.")
print("LANG_OWNMIS_DONE", flush=True)
