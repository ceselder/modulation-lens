#!/usr/bin/env python3
"""
Does the model actually WANT to emit {y} after each template?

We prefill the carrier in the assistant turn. If the model would not naturally produce it there, the
read is of an off-distribution "forced to say something I would not say" state -- and a wrapper can
shift such a state around freely. Prediction: templates where the carrier is UNLIKELY are the ones
that reward the pizza wrapper.

Per template, with the slot filled by neutral fillers:
  nll        mean per-token NLL of the carrier at the prefill position (LOW = the model wants it)
  p_first    probability of the carrier's FIRST token (the hardest one to get right)
  greedy     does greedy decoding reproduce the carrier's first 4 tokens
then correlate against the bare / bonus numbers already measured.
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()

INCUMBENT = dict(zip(["tpl%d" % i for i in range(6)], C.TEMPLATES_RECOVERED))
CANDS = json.load(open("/workspace/inv/results/template_search.json"))
NEW = {
 "incumbent-tpl0": C.TEMPLATES_RECOVERED[0],
 "y-first": 'Repeat this sentence exactly: "{y}"\n\nAs you write it, keep "{x}" at the front of '
            'your mind. Output only the sentence.',
 "roleplay": 'You are someone who cannot stop thinking about "{x}". Asked to read a line aloud, '
             'you say only: "{y}"',
 "constraint-strict": 'Think about "{x}".\n\nConstraint: your entire reply must be the string "{y}" '
                      'and nothing else. Do not explain. Do not add punctuation.',
 "assoc-prime": 'Priming context: "{x}"\n\nNow produce this exact string: "{y}"\n\nNo commentary.',
 "background": 'Background thought (do not mention it): "{x}"\n\nTask: output exactly this line, '
               'nothing else: "{y}"',
}
ALLT = {**INCUMBENT, **NEW}
CAR = C.CARRIERS_RECOVERED[0]
FILL = ["policy river garden engine", "lantern meadow cipher tunnel", "beacon quarry saddle",
        "vellum wharf pigment rafter"]
cid = tok(CAR, add_special_tokens=False).input_ids
print("[c] carrier %r = %d tokens\n" % (CAR, len(cid)), flush=True)

print("%-20s %8s %9s %8s   %s" % ("template", "nll", "p_first", "greedy", "first greedy tokens"))
res = {}
for name, tpl in ALLT.items():
    nlls, pf, gm, gtxt = [], [], [], ""
    for f in FILL:
        body = tpl.replace("{x}", f).replace("{y}", CAR)
        rend = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        pre = tok(rend, add_special_tokens=False).input_ids
        with torch.no_grad():
            out = model(input_ids=torch.tensor([pre + cid], device=dev))
            lg = torch.log_softmax(out.logits[0, :-1].float(), -1)
            tg = torch.tensor(pre + cid, device=dev)[1:]
            lo = len(pre) - 1
            lp = lg[lo:lo + len(cid)].gather(1, tg[lo:lo + len(cid)].unsqueeze(1)).squeeze(1)
            nlls.append(float(-lp.mean()))
            pf.append(float(lp[0].exp()))
            g = model.generate(input_ids=torch.tensor([pre], device=dev),
                               attention_mask=torch.ones(1, len(pre), device=dev, dtype=torch.long),
                               max_new_tokens=5, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            got = g[0, len(pre):].tolist()
            gm.append(got[:4] == cid[:4])
            if not gtxt:
                gtxt = tok.decode(got)
    res[name] = {"nll": float(np.mean(nlls)), "p_first": float(np.mean(pf)),
                 "greedy": float(np.mean(gm))}
    print("%-20s %8.3f %9.4f %8.0f%%   %r"
          % (name, res[name]["nll"], res[name]["p_first"], 100 * res[name]["greedy"], gtxt[:34]),
          flush=True)

known = {c["name"]: c for c in CANDS}
pairs = [(n, res[n]["nll"], known[n]["bare"], known[n]["bonus"])
         for n in res if n in known]
if len(pairs) >= 4:
    nl = np.array([p[1] for p in pairs]); ba = np.array([p[2] for p in pairs])
    bo = np.array([p[3] for p in pairs])
    print("\n=== correlation across %d templates with known scores ===" % len(pairs))
    print("  corr(carrier NLL, bare score)   = %+.3f   (expect NEGATIVE: wanted carrier -> good bare)"
          % float(np.corrcoef(nl, ba)[0, 1]))
    print("  corr(carrier NLL, wrapper bonus)= %+.3f   (expect POSITIVE: unwanted carrier -> hackable)"
          % float(np.corrcoef(nl, bo)[0, 1]))
    for n, a, b, c in sorted(pairs, key=lambda x: x[1]):
        print("    %-20s nll %6.3f  bare %.4f  bonus %+.4f" % (n, a, b, c))
json.dump(res, open("/workspace/inv/results/carrier_nll.json", "w"), indent=1)
print("\nCARRIER_NLL_DONE", flush=True)
