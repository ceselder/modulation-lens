#!/usr/bin/env python3
"""Is the lens READING the activation, or just fitting the dictionary's phrasebook?

Training perplexity on the four-bullet targets fell from ~24 to ~12 over one epoch. That is
ambiguous: the model could be learning the mapping activation -> bullets, or merely learning what
thinkies atom labels look like (their vocabulary, length, phrasings, co-occurrence). The second
lowers loss just as effectively and would leave the bullets uninformative about the specific
activation -- which the 23.7% atom-relevance figure makes a live possibility.

The discriminating measurement is a MISMATCH control, the same logic this project already uses for
`disc`: score the bullets mined for activation i while injecting a DIFFERENT activation j.

  matched loss  ~=  mismatched loss   -> the activation does no work; it is a phrasebook fit
  matched loss  <<  mismatched loss   -> the gap is the part that is genuinely a readout

Reports both, their gap in nats, and the implied perplexity ratio. Teacher-forced, no generation,
so it is cheap.
"""
import argparse, json, os, sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--bullets", default="/root/data/nnols4_500k.jsonl")
ap.add_argument("--data", default="/root/data/prose_L42_500k.parquet")
ap.add_argument("--n-pool", type=int, default=500000)
ap.add_argument("--n", type=int, default=512, help="activations scored")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--bullet-max-tok", type=int, default=10)
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--inject", default="karvonen")
ap.add_argument("--shift", type=int, default=101,
                help="mismatch offset: activation i is paired with bullets from i+shift, so the "
                     "mismatched set is the SAME set of bullets and the SAME set of activations, "
                     "just re-paired -- no distributional difference between the conditions")
ap.add_argument("--out", default="/workspace/inv/results/mismatch_control.jsonl")
A = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = base.model
INJ, LEFT, RIGHT = C.marker_ids(tok)
HOOK = {"vec": None, "ids": None}
inner.register_forward_pre_hook(
    lambda m, a, kw: HOOK.__setitem__("ids", kw.get("input_ids") if kw.get("input_ids") is not None
                                      else (a[0] if a else None)), with_kwargs=True)


def _inj(m, a, out):
    resid = out[0] if isinstance(out, tuple) else out
    ids, vec = HOOK["ids"], HOOK["vec"]
    if vec is None or ids is None or tuple(ids.shape) != tuple(resid.shape[:-1]):
        return out
    if not bool((ids == INJ).any()):
        return out
    new = C.inject_at_marker(ids, resid, vec, INJ, LEFT, RIGHT, mode=A.inject)
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


inner.layers[1].register_forward_hook(_inj)
model = PeftModel.from_pretrained(base, A.ckpt, adapter_name="m").eval()
model.set_adapter("m")

pf = os.path.join(A.ckpt, "prompt.txt")
if not os.path.exists(pf):
    pf = os.path.join(os.path.dirname(A.ckpt.rstrip("/")), "prompt.txt")
JOB = open(pf).read()
PROMPT = tok.apply_chat_template([{"role": "user", "content": JOB}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
PIDS = tok.encode(PROMPT, add_special_tokens=False)
assert PIDS.count(INJ) == 1

rows = [json.loads(l) for l in open(A.bullets)][: A.n]
acc = []
for b in pq.ParquetFile(A.data).iter_batches(batch_size=8192, columns=["activation_vector"]):
    acc.append(np.asarray(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in acc) >= A.n_pool:
        break
V = np.concatenate(acc)
del acc


def trunc(b):
    ids = tok(b, add_special_tokens=False).input_ids
    return tok.decode(ids[: A.bullet_max_tok]).strip() if len(ids) > A.bullet_max_tok else b


def target_ids(r):
    txt = "\n".join("* " + trunc(b) for b in r["bullets"])
    return tok(txt, add_special_tokens=False).input_ids[: A.max_new]


@torch.no_grad()
def mean_nll(pairs):
    """pairs: list of (activation_row_index, bullets_row). Teacher-forced mean NLL per token."""
    tot_nll, tot_tok = 0.0, 0
    for s in range(0, len(pairs), A.batch):
        chunk = pairs[s:s + A.batch]
        seqs = [PIDS + target_ids(r) for _, r in chunk]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(chunk), L), 248046, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        for k, (sq, (_, r)) in enumerate(zip(seqs, chunk)):
            ids[k, :len(sq)] = torch.tensor(sq)
            lab[k, len(PIDS):len(sq)] = torch.tensor(sq[len(PIDS):])
        HOOK["vec"] = torch.from_numpy(V[[i for i, _ in chunk]]).to(dev).float()
        try:
            lg = model(input_ids=ids.to(dev)).logits[:, :-1].float()
        finally:
            HOOK["vec"] = None
        tgt = lab[:, 1:].to(dev)
        m = tgt != -100
        nll = torch.nn.functional.cross_entropy(
            lg[m], tgt[m], reduction="sum")
        tot_nll += float(nll)
        tot_tok += int(m.sum())
    return tot_nll / max(tot_tok, 1), tot_tok


idx = [int(r["i"]) for r in rows]
matched = [(idx[k], rows[k]) for k in range(len(rows))]
# re-pair by a fixed shift: same activations, same bullets, only the pairing is broken
mis = [(idx[k], rows[(k + A.shift) % len(rows)]) for k in range(len(rows))]

mn, nt = mean_nll(matched)
xn, _ = mean_nll(mis)
out = {"ckpt": A.ckpt, "n": len(rows), "tokens_scored": nt,
       "matched_nll": round(mn, 4), "mismatched_nll": round(xn, 4),
       "gap_nats": round(xn - mn, 4),
       "matched_ppl": round(float(np.exp(mn)), 3), "mismatched_ppl": round(float(np.exp(xn)), 3),
       "ppl_ratio": round(float(np.exp(xn - mn)), 3)}
os.makedirs(os.path.dirname(A.out), exist_ok=True)
with open(A.out, "a") as f:
    f.write(json.dumps(out) + "\n")
print(json.dumps(out, indent=1), flush=True)
print("READ:", "activation is doing real work" if out["gap_nats"] > 0.15 else
      "NO readout -- loss is a phrasebook fit, the activation is nearly irrelevant", flush=True)
print("MISMATCH_CONTROL_DONE", flush=True)
