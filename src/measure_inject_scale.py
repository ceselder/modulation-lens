#!/usr/bin/env python3
"""How far off-scale is REPLACEMENT injection, vs Karvonen norm-matched addition?

We inject an L42 activation at decoder block 1. Residual norms grow with depth, so replacing a
block-1 residual with an L42 vector puts an out-of-scale magnitude into the stream; Karvonen's
h' = h + ||h||*v/||v|| takes only the DIRECTION of v and rescales to the local norm, which is why
the olens uses it. This measures the mismatch instead of arguing about it:

  - ||h|| at block 1 at the marker position (what replacement overwrites)
  - ||v||  of a real L42 activation (what replacement writes there)
  - the ratio, and what block-1 norms look like across the whole prompt for context
  - and, downstream, whether the two recipes even land in the same place: cosine between the
    block-42 readout state under replacement vs under Karvonen, for the same activation.

If the ratio is ~1 this is a non-issue. If it is far from 1, replacement is injecting a vector
whose magnitude belongs to a different depth, and a Karvonen-trained arm is worth running.
"""
import sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = model.model
INJ, LEFT, RIGHT = C.marker_ids(tok)

JOB = ("You are shown an internal activation vector captured from a language model at a single "
       "position while it was reading some text. The vector is enclosed in <concept> tags.\n\n"
       "<concept>%s</concept>\n\nYour job: write the short phrase that this state is holding in "
       "mind.\n\nOutput only the phrase." % C.INJ_CHAR)
txt = tok.apply_chat_template([{"role": "user", "content": JOB}], tokenize=False,
                              add_generation_prompt=True, enable_thinking=False)
ids = torch.tensor([tok.encode(txt, add_special_tokens=False)], device=dev)
mpos = int((ids[0] == INJ).nonzero().flatten()[0])
print("[i] prompt %d tok, marker at %d" % (ids.shape[1], mpos), flush=True)

GRAB = {}
inner.layers[1].register_forward_hook(
    lambda m, i, o: GRAB.__setitem__(1, (o[0] if isinstance(o, tuple) else o).detach().float()))
inner.layers[42].register_forward_hook(
    lambda m, i, o: GRAB.__setitem__(42, (o[0] if isinstance(o, tuple) else o).detach().float()))

with torch.no_grad():
    model(input_ids=ids)
h1 = GRAB[1][0]
n_marker = float(h1[mpos].norm())
n_all = h1.norm(dim=-1)
print("[i] block-1 residual norm: at marker %.2f | prompt mean %.2f median %.2f p5 %.2f p95 %.2f"
      % (n_marker, float(n_all.mean()), float(n_all.median()),
         float(n_all.quantile(0.05)), float(n_all.quantile(0.95))), flush=True)

V = []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=2048, columns=["activation_vector"]):
    V.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in V) >= 2048:
        break
A = torch.from_numpy(np.concatenate(V)[:64]).to(dev)
nv = float(A.norm(dim=1).mean())
print("[i] L42 activation norm (what we WRITE there): mean %.2f" % nv, flush=True)
print("[i] RATIO ||v|| / ||h_block1@marker|| = %.2f  -> replacement writes a vector %s"
      % (nv / max(n_marker, 1e-6),
         "%.1fx the local scale" % (nv / max(n_marker, 1e-6)) if nv > n_marker
         else "%.2fx the local scale" % (nv / max(n_marker, 1e-6))), flush=True)

# ---- do the two recipes even land in the same place? ----
def read42(vec, mode):
    box = {"v": vec, "mode": mode}

    def hk(m, i, o):
        resid = o[0] if isinstance(o, tuple) else o
        if resid.shape[1] != ids.shape[1]:
            return o
        new = resid.clone()
        v = box["v"].to(new.dtype)
        if box["mode"] == "replace":
            new[0, mpos] = v
        else:                                        # karvonen norm-match add
            h = resid[0, mpos]
            new[0, mpos] = h + h.norm() * (v / v.norm().clamp(min=1e-6))
        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new

    hd = inner.layers[1].register_forward_hook(hk)
    try:
        with torch.no_grad():
            model(input_ids=ids)
        return GRAB[42][0, -1].clone()
    finally:
        hd.remove()


cos, dn = [], []
for j in range(16):
    a = read42(A[j], "replace")
    b = read42(A[j], "karvonen")
    cos.append(float((a @ b) / (a.norm() * b.norm() + 1e-8)))
    dn.append(float(a.norm() / b.norm()))
print("[i] block-42 read state, replace vs karvonen (n=16): cos %.4f +-%.4f | norm ratio %.3f"
      % (float(np.mean(cos)), float(np.std(cos)), float(np.mean(dn))), flush=True)
print("[i] cos ~1 => the recipes are interchangeable; cos << 1 => they are different lenses and "
      "the trained one is the only valid one to generate with.")
print("INJECT_SCALE_DONE", flush=True)
