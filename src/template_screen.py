#!/usr/bin/env python3
"""
Large template screen. Gate A rebuilt: generate many, measure, keep the best.

Design rules carried in, from what actually replicated:
  * carrier must be maximally FORCED (low NLL) -- if it is a real choice the model spends capacity
    deciding what to write instead of holding {x}. Verified: incumbents sit at NLL 0.000-0.004.
  * bind {x} grammatically to the act of writing. The one mechanism that replicated: +0.046 over
    isolating it in its own sentence.
  * keep the buffer after {y} SHORT. Longer buffers measured worse (bare 0.218 -> 0.175) and more
    hackable (bonus +0.120 -> +0.226).

Scored on HELD-OUT blogpost positions (not the ones used to find y-first), on:
  bare   plain content phrase                  want HIGH
  disc   bare minus bare-against-other-targets want HIGH
  bonus  what the pizza wrapper still adds     want LOW
  nll    carrier NLL at the prefill            want LOW
"""
import itertools, json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

NCAR, LO, HI = 2, 40, 76          # held-out slice of the 96 scored positions
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
CARS = C.CARRIERS_RECOVERED[:NCAR]

HOLD = ['Focus on the following idea: "{x}"',
        'Keep "{x}" firmly in mind',
        'While preoccupied with "{x}"',
        'Holding "{x}" in your thoughts',
        'With "{x}" foremost in your attention',
        'Dwelling on "{x}"',
        'Attending closely to "{x}"',
        'Fixated on "{x}"']
SAY = ['write the following phrase exactly: "{y}"',
       'repeat this line verbatim: "{y}"',
       'copy out exactly this: "{y}"',
       'reproduce this sentence word for word: "{y}"']
TAIL = ['', ' Output only that.', ' Nothing else.']
CAND = {}
for i, (h, sy) in enumerate(itertools.product(HOLD, SAY)):
    for j, t in enumerate(TAIL):
        CAND["h%d.s%d.t%d" % (HOLD.index(h), SAY.index(sy), j)] = h + ", " + sy + t
# y-first variants of the same, since y-first tied the incumbent
for i, (h, sy) in enumerate(itertools.product(HOLD[:4], SAY[:2])):
    CAND["Y.h%d.s%d" % (HOLD.index(h), SAY.index(sy))] = (
        sy[0].upper() + sy[1:] + "\n\n" + "As you write it, " +
        h[0].lower() + h[1:] + ". Output only the sentence.")
CAND["REF-incumbent-tpl0"] = C.TEMPLATES_RECOVERED[0]
CAND["REF-y-first"] = ('Repeat this sentence exactly: "{y}"\n\nAs you write it, keep "{x}" at the '
                       'front of your mind. Output only the sentence.')
CAND["REF-weak-tpl2"] = C.TEMPLATES_RECOVERED[2]
print("[s] %d candidates x %d carriers, held-out positions %d-%d" % (len(CAND), NCAR, LO, HI),
      flush=True)

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
keys = list(g350)[LO:HI]
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
FILL = [" ".join(rr.choice(words) for _ in range(rr.randint(3, 9))) for _ in range(32)]
ALL = sorted(set(CORE.values()) | set(FULL.values()) | set(FILL))
cid = tok(CARS[0], add_special_tokens=False).input_ids

rows = []
for n, (name, tpl) in enumerate(CAND.items()):
    G = C.Grid(tok, [tpl], CARS, 42, J, dev)
    bm, bx, pm = [], [], []
    with torch.no_grad():
        for ci in range(NCAR):
            rv = G.read(model, ALL, L42, carrier=ci, max_tok=32)
            pmu = torch.stack([rv[f] for f in FILL]).mean(0)
            u = {s: (rv[s] - pmu) / (rv[s] - pmu).norm().clamp(min=1e-8) for s in ALL}
            for k in keys:
                bm.append(float(u[CORE[k]] @ TGT[k]))
                pm.append(float(u[FULL[k]] @ TGT[k]))
                bx += [float(u[CORE[k]] @ TGT[k2]) for k2 in keys[:8] if k2 != k]
        body = tpl.replace("{x}", FILL[0]).replace("{y}", CARS[0])
        rend = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        pre = tok(rend, add_special_tokens=False).input_ids
        o = model(input_ids=torch.tensor([pre + cid], device=dev))
        lg = torch.log_softmax(o.logits[0, :-1].float(), -1)
        tgv = torch.tensor(pre + cid, device=dev)[1:]
        lo_ = len(pre) - 1
        nll = float(-lg[lo_:lo_ + len(cid)].gather(
            1, tgv[lo_:lo_ + len(cid)].unsqueeze(1)).mean())
    r = {"name": name, "tpl": tpl, "bare": float(np.mean(bm)), "mis": float(np.mean(bx)),
         "pizza": float(np.mean(pm)), "nll": nll}
    r["bonus"] = r["pizza"] - r["bare"]
    r["disc"] = r["bare"] - r["mis"]
    r["score"] = r["disc"] - r["bonus"]
    rows.append(r)
    if n % 10 == 0 or name.startswith("REF"):
        print("  [%3d/%d] %-20s bare %.4f disc %.4f bonus %+.4f nll %.4f"
              % (n, len(CAND), name, r["bare"], r["disc"], r["bonus"], r["nll"]), flush=True)

rows.sort(key=lambda r: -r["score"])
print("\n=== top 12 by (disc - bonus), held-out positions ===")
print("%-20s %7s %7s %8s %7s %7s" % ("name", "bare", "disc", "bonus", "nll", "score"))
for r in rows[:12]:
    print("%-20s %7.4f %7.4f %+8.4f %7.4f %7.4f"
          % (r["name"], r["bare"], r["disc"], r["bonus"], r["nll"], r["score"]))
print("\n=== references ===")
for r in rows:
    if r["name"].startswith("REF"):
        print("%-20s %7.4f %7.4f %+8.4f %7.4f %7.4f  (rank %d)"
              % (r["name"], r["bare"], r["disc"], r["bonus"], r["nll"], r["score"],
                 rows.index(r) + 1))
print("\n=== best template, verbatim ===")
print(repr(rows[0]["tpl"]))
json.dump(rows, open("/workspace/inv/results/template_screen.json", "w"), indent=1)
print("\nSCREEN_DONE", flush=True)
