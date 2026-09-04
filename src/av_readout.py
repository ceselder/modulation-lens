#!/usr/bin/env python3
"""What does the SFT'd AV actually SAY for held-out activations?

The gate that matters. ppl went 502 -> 8.5, but this project has measured twice that SFT loss is
not a proxy for readout quality (ppl 24->17 moved the probe 1.5 sigma the WRONG way; NNOLS
warm-starting dropped it 0.433 -> 0.316). So read the text.

Recipe copied from inv_train rather than reinvented:
  * the prompt is the VERBATIM prompt.txt saved beside the adapter -- a lens must be read with the
    prompt it was trained on, and a stale shared prompt is a known way to get malformed readouts
    that look like a bad lens
  * the activation is injected at layers[1] (NOT the read layer) at the <concept> marker
  * inject mode must match training: 'replace'
"""
import argparse, json, os, sys
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inv_core as C

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--base", default="Qwen/Qwen3.6-27B")
p.add_argument("--probe-npy", required=True)
p.add_argument("--probe-meta", default="")
p.add_argument("--n", type=int, default=24)
p.add_argument("--max-new", type=int, default=24)
p.add_argument("--temp", type=float, default=0.0)
p.add_argument("--inject", default="replace")
p.add_argument("--layer", type=int, default=42)
p.add_argument("--out", default="")
A = p.parse_args()

dev = "cuda"
tok = AutoTokenizer.from_pretrained(A.base)
m = AutoModelForCausalLM.from_pretrained(A.base, dtype=torch.bfloat16).to(dev).eval()
inner = m.model
model = PeftModel.from_pretrained(m, A.ckpt).eval()
n_lora = sum(1 for k, _ in model.named_parameters() if "lora" in k)
print("[load] adapter %s | %d lora tensors" % (A.ckpt, n_lora), flush=True)
if n_lora == 0:
    raise SystemExit("adapter loaded 0 LoRA tensors -- wrong dir or a naming mismatch")

INJ, LEFT, RIGHT = C.marker_ids(tok)
HOOK = {"ids": None, "vec": None}


def _stash(mod, args, kwargs):
    HOOK["ids"] = kwargs.get("input_ids", args[0] if args else None)


def _inject(mod, a, out):
    resid = out[0] if isinstance(out, tuple) else out
    ids, vec = HOOK["ids"], HOOK["vec"]
    if vec is None or ids is None or tuple(ids.shape) != tuple(resid.shape[:-1]):
        return out
    if not bool((ids == INJ).any()):
        return out
    new = C.inject_at_marker(ids, resid, vec, INJ, LEFT, RIGHT, A.inject)
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


inner.register_forward_pre_hook(_stash, with_kwargs=True)
inner.layers[1].register_forward_hook(_inject)

# The VERBATIM job text the adapter was trained with -- but inv_train saves JOB (raw), NOT the
# chat-wrapped PROMPT_TXT it encodes, so the template MUST be re-applied here with the same kwargs.
# Encoding the raw text gave 174 tokens vs training's 186, and with no assistant-turn framing
# greedy decoding emitted EOS immediately: every readout came back as ''.
_d = A.ckpt if os.path.exists(os.path.join(A.ckpt, "prompt.txt")) \
    else os.path.dirname(A.ckpt.rstrip("/"))
job = open(os.path.join(_d, "prompt.txt")).read()
ptxt = tok.apply_chat_template([{"role": "user", "content": job}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
PIDS = torch.tensor(tok.encode(ptxt, add_special_tokens=False), device=dev)
PLEN = PIDS.shape[0]
_at = (PIDS == INJ).nonzero().flatten()
print("[prompt] %d tokens | markers found: %d | marker at %s"
      % (PLEN, _at.numel(), _at.tolist()[:3]), flush=True)
assert _at.numel() == 1, "prompt needs exactly one marker, found %d" % _at.numel()
assert int(PIDS[int(_at[0]) - 1]) == LEFT and int(PIDS[int(_at[0]) + 1]) == RIGHT, \
    "marker neighbours wrong -- the injection would land in the wrong place"

ACT = torch.from_numpy(np.load(A.probe_npy).astype("float32"))[: A.n]
META = [json.loads(l) for l in open(A.probe_meta)][: A.n] if A.probe_meta else []
print("[probe] %d activations from %s\n" % (ACT.shape[0], A.probe_npy), flush=True)

rows = []
with torch.no_grad():
    for s in range(0, ACT.shape[0], 8):
        sub = ACT[s:s + 8].to(dev)
        B = sub.shape[0]
        HOOK["vec"] = sub
        try:
            gen = model.generate(
                input_ids=PIDS.unsqueeze(0).expand(B, -1).contiguous(),
                attention_mask=torch.ones(B, PLEN, device=dev, dtype=torch.long),
                max_new_tokens=A.max_new, do_sample=A.temp > 0,
                temperature=A.temp if A.temp > 0 else None,
                top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
        finally:
            HOOK["vec"] = None
        for j, t in enumerate(tok.batch_decode(gen[:, PLEN:], skip_special_tokens=True)):
            i = s + j
            ctx = (META[i].get("ctx", "") if i < len(META) else "")[-90:]
            mark = (META[i].get("mark", "") if i < len(META) else "")
            rows.append({"i": i, "readout": t.strip(), "mark": mark, "ctx": ctx})
            print("  [%02d] mark=%r\n       ctx ...%s\n       LENS SAYS: %r"
                  % (i, mark[:44], ctx.replace("\n", " "), t.strip()[:90]), flush=True)
if A.out:
    json.dump(rows, open(A.out, "w"), indent=1)
print("\nAV_READOUT_DONE", flush=True)
