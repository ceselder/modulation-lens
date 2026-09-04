#!/usr/bin/env python3
"""
Blogpost punctuation positions as the RL held-out set.

Why: the previous held-out was 32 random FineWeb activations, whose readouts are meaningless to a
human -- so neither the reward number nor the sampled phrases told us whether training was working.
Punctuation in a text you wrote is interpretable: you know what the sentence was about, so a phrase
is checkable by eye. Punctuation specifically because those are the positions where the model has
just finished a thought and is summarising it.

Read CHAT-NATIVELY (paragraph inside a user turn), because raw-vs-chat reads differ by whitened
cosine 0.75 averaged over positions and as low as 0.06 at the worst -- a raw read describes a state
the model never occupies.
"""
import argparse, collections, glob, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--text-file", default="/workspace/cnla/skip-lens/data/blogpost.txt")
ap.add_argument("--layer", type=int, default=42)
ap.add_argument("--n", type=int, default=32, help="positions to keep, stratified by mark")
ap.add_argument("--max-tok", type=int, default=256)
ap.add_argument("--out", default="/workspace/cnla/skip-lens/data/sft_invert/holdout_blogpost")
A = ap.parse_args()

BASE = "Qwen/Qwen3.6-27B"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
G = {}
model.model.layers[A.layer].register_forward_hook(
    lambda m, i, o: G.__setitem__("h", o[0] if isinstance(o, tuple) else o))

_r = tok.apply_chat_template([{"role": "user", "content": "XSLOT"}], tokenize=False,
                             add_generation_prompt=True, enable_thinking=False)
_a, _b = _r.split("XSLOT")
CPRE = tok(_a, add_special_tokens=False).input_ids
CPOST = tok(_b, add_special_tokens=False).input_ids
print("[hb] chat wrapper: %d + text + %d tokens" % (len(CPRE), len(CPOST)), flush=True)

DELIM = {}
for s in ". , ! ? ; : ...".split():
    for c in (s, " " + s):
        i = tok(c, add_special_tokens=False).input_ids
        if len(i) == 1:
            DELIM[i[0]] = s

paras = [x.strip() for x in open(A.text_file).read().split("\n\n") if x.strip()]
cand = []
with torch.no_grad():
    for pi, para in enumerate(paras):
        ids = tok(para, add_special_tokens=False, truncation=True, max_length=A.max_tok).input_ids
        if not ids:
            continue
        model(input_ids=torch.tensor([CPRE + ids + CPOST], device="cuda"))
        Hh = G["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
        strs = [tok.decode([i]) for i in ids]
        for k in range(len(ids)):
            if ids[k] in DELIM:
                cand.append({"para": pi, "i": k, "mark": DELIM[ids[k]],
                             "tok": strs[k],
                             "ctx": "".join(strs[max(0, k - 22):k + 1]),
                             "vec": Hh[k].cpu().numpy()})
print("[hb] %d punctuation positions found" % len(cand), flush=True)

# stratify by mark so the eval is not 200 full stops
by = collections.defaultdict(list)
for c in cand:
    by[c["mark"]].append(c)
order = sorted(by, key=lambda m: -len(by[m]))
pick, r = [], A.n
while r > 0 and any(by[m] for m in order):
    for m in order:
        if not by[m] or r <= 0:
            continue
        # spread across the document rather than taking the first ones
        step = max(1, len(by[m]) // max(1, r))
        pick.append(by[m].pop(min(step, len(by[m]) - 1) // 1 * 0))
        r -= 1
pick = pick[:A.n]
np.save(A.out + ".npy", np.stack([p["vec"] for p in pick]).astype("float32"))
with open(A.out + ".jsonl", "w") as f:
    for p in pick:
        f.write(json.dumps({k: v for k, v in p.items() if k != "vec"}) + "\n")
print("[hb] kept %d positions: %s" % (len(pick),
      dict(collections.Counter(p["mark"] for p in pick))), flush=True)
for p in pick[:10]:
    print("   p%-3d i%-4d %r  ...%s" % (p["para"], p["i"], p["tok"], p["ctx"][-58:]))
print("\nwrote %s.npy / .jsonl\nHARVEST_HOLDOUT_DONE" % A.out, flush=True)
