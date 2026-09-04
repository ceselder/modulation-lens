#!/usr/bin/env python3
"""Collect label / J-lens / MAEMM readouts for N thinkies vectors, stratified by reliability.

Generation only -- scoring and plotting happen on CPU afterwards, so the GPU is not held while an
LLM judge runs. Saves raw texts so the judged metric is reproducible without regenerating.
"""
import argparse, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

p = argparse.ArgumentParser()
p.add_argument("--n-per-bucket", type=int, default=60)
p.add_argument("--samples", type=int, default=3)
p.add_argument("--topk", type=int, default=12)
p.add_argument("--max-new", type=int, default=40)
p.add_argument("--out", default="/workspace/inv/results/readouts_for_judge.json")
A = p.parse_args()
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
MP, VREF = list(_p) + tok.encode(MARKER, add_special_tokens=False), {"v": None}
MPOS = len(MP) - 1


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
rng = np.random.default_rng(11)
BUCKETS = (("confused", 0.65, 0.70), ("middling", 0.70, 0.78), ("clear", 0.78, 1.01))
rows = []
for tag, lo, hi in BUCKETS:
    idx = np.where((R >= lo) & (R < hi))[0]
    sel = rng.choice(idx, min(A.n_per_bucket, len(idx)), replace=False)
    print("[c] %-9s n=%d reliability %.3f-%.3f" % (tag, len(sel), R[sel].min(), R[sel].max()),
          flush=True)
    for i in sel:
        rows.append({"i": int(i), "bucket": tag, "reliability": float(R[i]), "label": labs[i]})

B = 8
with torch.no_grad():
    for s in range(0, len(rows), B):
        chunk = rows[s:s + B]
        vt = torch.from_numpy(V[[r["i"] for r in chunk]].astype("float32")).to(dev)
        lg = (vt @ J.T) @ WU.T.float()
        for k, r in enumerate(chunk):
            r["jlens"] = " ".join(tok.decode([j]).strip()
                                  for j in torch.topk(lg[k], A.topk).indices.tolist())
        outs = [[] for _ in chunk]
        for _ in range(A.samples):
            ids = torch.tensor([MP] * len(chunk), device=dev)
            VREF["v"] = vt
            try:
                model.set_adapter("maemm")
                g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   do_sample=True, temperature=0.9, top_p=0.95,
                                   max_new_tokens=A.max_new, pad_token_id=EOS)
            finally:
                VREF["v"] = None
            for k, row in enumerate(g[:, len(MP):].tolist()):
                cut = row.index(EOS) if EOS in row else len(row)
                outs[k].append(tok.decode(row[:cut], skip_special_tokens=True).strip())
        for k, r in enumerate(chunk):
            r["maemm"] = outs[k]
        if s % 40 == 0:
            print("[c] %d/%d" % (s, len(rows)), flush=True)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(rows, open(A.out, "w"), indent=1)
print("[c] wrote %d rows -> %s" % (len(rows), A.out), flush=True)
print("COLLECT_DONE", flush=True)
