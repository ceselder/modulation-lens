#!/usr/bin/env python3
"""Side-by-side readouts for the LEAST-reliable atoms available: label vs J-lens vs MAEMM.

Note the ceiling on "confusing": thinkies-v3 was filtered at reliability >= 0.65 at harvest, so the
bottom of this dataset is only mildly ambiguous. Also dumps a few SYNTHETIC mixtures --
unit(v_a + v_b) for two unrelated atoms -- which are genuinely ambiguous by construction and have a
known ground truth (both ingredients), to show what a real test would look like.
"""
import glob, json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

N_LOW, N_MIX, SAMP, TOPK, MAXNEW = 0, 0, 3, 12, 40
dev, EOS = "cuda", 248046
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
J = C.load_jlens(42, dev)
WU = base.lm_head.weight.detach()

MARKER = " ?"
INSTR = ("You are an interpretability research tool. Researchers have injected a single linear probe "
         "direction from this model's own layer-42 residual stream. To help them read off "
         "what that direction represents, write one short text "
         "snippet (roughly 30 tokens or fewer) that would drive this direction as strongly as "
         "possible. Output only the snippet itself, with no explanation, preamble, or quotation "
         "marks. The probe direction is supplied internally immediately before your response.")
_p = tok.apply_chat_template([{"role": "user", "content": INSTR}], tokenize=True,
                             add_generation_prompt=True, enable_thinking=False)
_p = _p["input_ids"] if hasattr(_p, "keys") else _p
while isinstance(_p[0], list):
    _p = _p[0]
MP = list(_p) + tok.encode(MARKER, add_special_tokens=False)
MPOS = len(MP) - 1
VREF = {"v": None}


def _hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    v = VREF["v"]
    if v is None or h.shape[1] <= 1:
        return out
    b = h[:, MPOS]
    h[:, MPOS] = b + (torch.nn.functional.normalize(v, dim=-1)
                      * b.norm(dim=-1, keepdim=True)).to(h.dtype)
    return out


base.model.layers[1].register_forward_hook(_hook)
model = PeftModel.from_pretrained(base, "ceselder/qwen36-27b-maemm-inverter",
                                  adapter_name="maemm").eval()

labs, vecs, rels = [], [], []
for sh in sorted(glob.glob("/workspace/thinkies/v3/thinkies_v3-*.parquet")):
    for b in pq.ParquetFile(sh).iter_batches(batch_size=16384,
                                            columns=["label", "vector", "reliability"]):
        l = b.column("label").to_pylist(); labs += l
        vecs.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                               dtype="float32").reshape(len(l), -1))
        rels += list(np.asarray(b.column("reliability").to_numpy(zero_copy_only=False),
                               dtype="float32"))
    if len(labs) >= 250000:
        break
V, R = np.concatenate(vecs), np.asarray(rels)
order = np.argsort(R)


@torch.no_grad()
def readouts(v, tag, extra=""):
    vt = torch.from_numpy(v.astype("float32")).to(dev).unsqueeze(0)
    lg = (vt @ J.T) @ WU.T.float()
    jl = None
    outs = []
    for _ in range(SAMP):
        ids = torch.tensor([MP], device=dev)
        VREF["v"] = vt
        try:
            model.set_adapter("maemm")
            g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                               temperature=0.9, top_p=0.95, max_new_tokens=MAXNEW,
                               pad_token_id=EOS)
        finally:
            VREF["v"] = None
        row = g[0, len(MP):].tolist()
        cut = row.index(EOS) if EOS in row else len(row)
        outs.append(tok.decode(row[:cut], skip_special_tokens=True).strip().replace("\n", " / "))
    print("\n%s %s" % (tag, extra), flush=True)
    for k, o in enumerate(outs):
        print("  MAEMM%d : %s" % (k + 1, o[:150]), flush=True)


print("=" * 100)
print("MAEMM applied DIRECTLY to thinkies modulation vectors (raw layer-42 direction, unit-normed,")
print("MAEMM's own prompt/marker/norm-matched injection). Spread across the reliability range.")
print("=" * 100)
rng0 = np.random.default_rng(7)
picks = []
for lo, hi, tag in ((0.65, 0.70, "low-agreement"), (0.70, 0.78, "mid"), (0.78, 1.01, "high")):
    idx = np.where((R >= lo) & (R < hi))[0]
    picks += [(i, tag) for i in rng0.choice(idx, 8, replace=False)]
for i, tag in picks:
    readouts(V[i], "LABEL  : %r" % labs[i][:70], "   [%s, reliability %.3f]" % (tag, R[i]))

print("\n" + "=" * 100)
print("SYNTHETIC MIXTURES  unit(v_a + v_b) of two unrelated atoms -- genuinely ambiguous, and the")
print("ground truth is KNOWN (both ingredients). This is what a real test of the claim needs.")
print("=" * 100)
rng = np.random.default_rng(0)
hi = order[-40000:]
for _ in range(N_MIX):
    a, b_ = rng.choice(hi, 2, replace=False)
    mix = V[a] / np.linalg.norm(V[a]) + V[b_] / np.linalg.norm(V[b_])
    readouts(mix, "INGREDIENTS: %r  +  %r" % (labs[a][:40], labs[b_][:40]))
print("\nDUMP_DONE", flush=True)
