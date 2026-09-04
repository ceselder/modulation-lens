#!/usr/bin/env python3
"""
Find templates where BARE content already works, so the abstraction wrapper buys little.

Observed on the 6 surviving templates: bare score and wrapper bonus are inversely related. The one
structurally distinct template (tpl0, the only one with a separate explicit output constraint) has
the best bare score (0.321 vs 0.18-0.25) and the smallest bonus (0.045). The other five are near
paraphrases -- concentrate / think deeply / deeply reflect / keep thinking / hold the concept -- and
reward the wrapper heavily.

So vary STRUCTURE, not the verb. Score each candidate on three things:
  bare_matched     how well a plain content phrase scores          (want HIGH)
  bare_mismatched  the same phrase against other positions          (want LOW -- discrimination)
  bonus            what the 'not pizza related wording but' wrapper still adds (want LOW)
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

N, NCAR = 20, 2
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
CARS = C.CARRIERS_RECOVERED[:NCAR]

# --- candidates. tpl0 is the incumbent that already works; the rest vary STRUCTURE:
#     explicit output constraints, separate sentences, order, framing of "hold in mind",
#     and register -- not another synonym for "think about".
CAND = {
 "incumbent-tpl0": 'Focus on the following idea: "{x}" while writing the following phrase: "{y}"\n\n'
                   "Write only the last phrase. Don't respond with anything else",
 "constraint-2sent": 'Your mind is occupied by this: "{x}"\n\nNow transcribe exactly this and '
                     'nothing more: "{y}"',
 "verbatim-copy": 'While preoccupied with "{x}", copy the following text verbatim: "{y}"\n\n'
                  'Output only the copied text.',
 "y-first": 'Repeat this sentence exactly: "{y}"\n\nAs you write it, keep "{x}" at the front of '
            'your mind. Output only the sentence.',
 "background": 'Background thought (do not mention it): "{x}"\n\nTask: output exactly this line, '
               'nothing else: "{y}"',
 "distraction": 'You are distracted by "{x}". Despite that, write out this sentence and stop: "{y}"',
 "assoc-prime": 'Priming context: "{x}"\n\nNow produce this exact string: "{y}"\n\nNo commentary.',
 "roleplay": 'You are someone who cannot stop thinking about "{x}". Asked to read a line aloud, '
             'you say only: "{y}"',
 "dual-task": 'Hold two things at once. In mind: "{x}". On the page: "{y}".\n\nWrite only what is '
              'on the page.',
 "terse": '"{x}" in mind. Write: "{y}". Nothing else.',
 "qa": 'Q: With "{x}" occupying your attention, what line were you asked to write?\nA: "{y}"',
 "annotate": 'Annotation: the writer was thinking about "{x}".\nText: "{y}"\n\nReproduce the text '
             'only.',
 "constraint-strict": 'Think about "{x}".\n\nConstraint: your entire reply must be the string "{y}" '
                      'and nothing else. Do not explain. Do not add punctuation.',
 "recall": 'Earlier you were told to remember "{x}". Now write down the sentence you were given: '
           '"{y}"\n\nJust the sentence.',
 "emotional": 'Something is on your mind: "{x}". You set it aside and neutrally write: "{y}"\n\n'
              'Output the sentence alone.',
}

V = []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=4096, columns=["activation_vector"]):
    V.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in V) >= 20000:
        break
AMU = torch.from_numpy(np.concatenate(V)[:20000]).mean(0).to(dev) @ J.T
d = json.load(open("/workspace/inv/results/blogpost_readouts.json"))
k_ = lambda r: (r["para"], r["i"])
g350 = {k_(r): r for r in d["runs"]["rl350"]["rows"]}
keys = list(g350)[:N]
paras = [x.strip() for x in open("/workspace/inv/data/blogpost.txt").read().split("\n\n")
         if x.strip()]
CPRE, CPOST = C.chat_wrap_ids(tok)
TGT = {}
with torch.no_grad():
    for k in keys:
        pi, ti = k
        ids = tok(paras[pi], add_special_tokens=False, truncation=True, max_length=256).input_ids
        ti = min(ti, len(ids) - 1)
        model(input_ids=torch.tensor([CPRE + ids + CPOST], device=dev))
        H = L42["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
        t = (H[ti] @ J.T) - AMU
        TGT[k] = t / t.norm().clamp(min=1e-8)
PRE = "not pizza related wording but related"
CORE = {k: (g350[k]["phrase"].replace(PRE, "").strip() or "the") for k in keys}
FULL = {k: g350[k]["phrase"] for k in keys}
import random
rr = random.Random(0)
words = ("policy river garden engine harbour lantern meadow cipher tunnel orchard beacon quarry "
         "saddle thistle vellum wharf pigment rafter cistern bramble").split()
FILL = [" ".join(rr.choice(words) for _ in range(rr.randint(3, 9))) for _ in range(40)]
ALL = sorted(set(CORE.values()) | set(FULL.values()) | set(FILL))
print("[t] %d candidates x %d carriers | %d positions | %d strings"
      % (len(CAND), NCAR, len(keys), len(ALL)), flush=True)

rows = []
for name, tpl in CAND.items():
    G = C.Grid(tok, [tpl], CARS, 42, J, dev)
    bm, bx, pm = [], [], []
    with torch.no_grad():
        for ci in range(NCAR):
            rv = G.read(model, ALL, L42, carrier=ci, max_tok=32)
            pmu = torch.stack([rv[f] for f in FILL]).mean(0)
            unit = {s: (rv[s] - pmu) / (rv[s] - pmu).norm().clamp(min=1e-8) for s in ALL}
            for i, k in enumerate(keys):
                bm.append(float(unit[CORE[k]] @ TGT[k]))
                pm.append(float(unit[FULL[k]] @ TGT[k]))
                bx += [float(unit[CORE[k]] @ TGT[k2]) for k2 in keys[:8] if k2 != k]
    r = {"name": name, "bare": float(np.mean(bm)), "mis": float(np.mean(bx)),
         "pizza": float(np.mean(pm))}
    r["bonus"] = r["pizza"] - r["bare"]
    r["disc"] = r["bare"] - r["mis"]
    rows.append(r)
    print("  %-20s bare %.4f  mis %+.4f  disc %.4f  bonus %+.4f"
          % (name, r["bare"], r["mis"], r["disc"], r["bonus"]), flush=True)

print("\n=== ranked by discrimination minus wrapper-dependence (disc - bonus) ===")
rows.sort(key=lambda r: -(r["disc"] - r["bonus"]))
print("%-20s %8s %8s %8s %8s" % ("template", "bare", "disc", "bonus", "score"))
for r in rows:
    print("%-20s %8.4f %8.4f %+8.4f %8.4f"
          % (r["name"], r["bare"], r["disc"], r["bonus"], r["disc"] - r["bonus"]))
json.dump(rows, open("/workspace/inv/results/template_search.json", "w"), indent=1)
print("\nTEMPLATE_SEARCH_DONE", flush=True)
