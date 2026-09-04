#!/usr/bin/env python3
"""
GO/NO-GO: does the verbatim preceding text work as an SFT label under the modulation reward?

The label is the text the model had just read at the target position. The reward asks a different
question: place a phrase in an instruction slot, force the model to write an unrelated carrier
sentence, and read its state there. Those are different substrates -- prose position versus
instruction slot -- so it is NOT obvious the causing text scores well on its own activation.

Test: score each position's OWN preceding text (matched) against other positions' preceding text
(mismatched). If matched >> mismatched the label carries signal and is worth an SFT epoch.

Also computes PMU empirically for THIS grid. The published thinkies ref_mean belongs to the old
16-template grid whose exact strings are lost, so reusing it would centre against the wrong thing.
"""
import glob, itertools, json, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, LAYER = "Qwen/Qwen3.6-27B", 42
random.seed(0)
torch.manual_seed(0)

# 6 templates recovered verbatim from the session transcript. The other 10 of the original 16 are
# lost with the box, so this is a NEW grid and its numbers are not comparable to the old ones.
TPL = [
 'Focus on the following idea: "{x}" while writing the following phrase: "{y}"\n\nWrite only the last phrase. Don\'t respond with anything else',
 'concentrate on "{x}" while you write "{y}". Respond only with "{y}", nothing else.',
 'Think deeply about "{x}" while replying with nothing but "{y}".',
 'Deeply reflect on "{x}" while responding with just the following sentence: "{y}".',
 'Keep thinking of "{x}" as you compose your reply, limiting it strictly to "{y}".',
 'Hold the concept of "{x}" in your thoughts, then respond with just "{y}".',
]
CARS = ['The chair stood near the window in the room.',
        'A small clock ticked on the shelf quietly.',
        'The book rested on the wooden table.',
        'Rain fell softly against the glass window.']

PASSAGES = [
 "The committee met for three hours before reaching any conclusion. Most of the discussion "
 "concerned the budget shortfall, which had grown steadily since the spring. Several members "
 "argued that the reserve fund should be tapped immediately, while others insisted that raising "
 "dues was the only sustainable path. By the end nobody was satisfied with the compromise.",
 "She had been repairing clocks for forty years and could tell what was wrong by sound alone. "
 "The trick, she said, was never to force a spring back into place; you waited until the metal "
 "was ready. Her workshop smelled of oil and brass shavings, and the walls were covered in "
 "movements she had never managed to finish.",
 "Coastal erosion along this stretch has accelerated markedly over the past decade. Sediment that "
 "once replenished the beaches is now trapped behind upstream dams, and the seawall built in the "
 "nineties has begun to undercut. Local authorities have proposed managed retreat, which remains "
 "deeply unpopular with property owners.",
]

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
G = {}
model.model.layers[LAYER].register_forward_hook(
    lambda m, i, o: G.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = torch.load(glob.glob("/workspace/.hf_home/hub/models--camilablank--workspace-lenses/"
                         "snapshots/*/qwen3.6-27b/j-lens/lens.pt")[0],
               map_location="cpu", weights_only=False)["J"][LAYER].to("cuda").float()

_r = tok.apply_chat_template([{"role": "user", "content": "XSLOT"}], tokenize=False,
                             add_generation_prompt=True, enable_thinking=False)
_a, _b = _r.split("XSLOT")
CPRE = tok(_a, add_special_tokens=False).input_ids
CPOST = tok(_b, add_special_tokens=False).input_ids

CELLS = []
for car in CARS:
    for t_ in TPL:
        body = t_.replace("{x}", "XSLOT").replace("{y}", "ZSLOT")
        rend = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        a_, b_ = rend.replace("ZSLOT", car).split("XSLOT")
        cid = tok(car, add_special_tokens=False).input_ids
        CELLS.append({"pre": tok(a_, add_special_tokens=False).input_ids,
                      "post": tok(b_, add_special_tokens=False).input_ids + cid,
                      "ncar": len(cid)})
print("[gn] grid: %d templates x %d carriers = %d cells" % (len(TPL), len(CARS), len(CELLS)),
      flush=True)


@torch.no_grad()
def read_slot(s):
    ids = tok(s, add_special_tokens=False).input_ids[:20] or tok(" the").input_ids
    acc = None
    for S in CELLS:
        model(input_ids=torch.tensor([S["pre"] + ids + S["post"]], device="cuda"))
        v = G["h"].float()[0, -S["ncar"]:, :].mean(0) @ J.T
        acc = v if acc is None else acc + v
    return acc / len(CELLS)


# --- PMU for THIS grid: average read over random filler phrases ---
VOCABW = [w for w in ("policy river garden engine harbour lantern meadow cipher tunnel orchard "
                      "beacon quarry saddle thistle vellum wharf").split()]
print("[gn] computing PMU over 24 random phrases for this grid...", flush=True)
acc = None
with torch.no_grad():
    for i in range(24):
        ph = " ".join(random.choice(VOCABW) for _ in range(random.randint(3, 8)))
        v = read_slot(ph)
        acc = v if acc is None else acc + v
PMU = acc / 24
print("[gn] |PMU| %.2f" % float(PMU.norm()), flush=True)

# --- harvest (activation, preceding text) pairs, chat-native ---
PAIRS = []
with torch.no_grad():
    for pi, p in enumerate(PASSAGES):
        ids = tok(p, add_special_tokens=False).input_ids[:256]
        model(input_ids=torch.tensor([CPRE + ids + CPOST], device="cuda"))
        H = G["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
        for k in (len(ids) // 4, len(ids) // 2, (3 * len(ids)) // 4, len(ids) - 1):
            PAIRS.append({"p": pi, "k": k, "h": H[k].clone(),
                          "label": tok.decode(ids[max(0, k - 15):k + 1]).strip()})
print("[gn] %d (activation, preceding-text) pairs\n" % len(PAIRS), flush=True)

AMU = torch.stack([q["h"] for q in PAIRS]).mean(0) @ J.T
LV = {}
with torch.no_grad():
    for q in PAIRS:
        LV[(q["p"], q["k"])] = read_slot(q["label"])


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


mat, mis = [], []
print("%-52s %8s %8s" % ("position's own preceding text (label)", "matched", "best-mis"))
for q in PAIRS:
    t = (q["h"] @ J.T) - AMU
    own = cos(LV[(q["p"], q["k"])] - PMU, t)
    others = [cos(LV[k] - PMU, t) for k in LV if k != (q["p"], q["k"])]
    mat.append(own)
    mis.extend(others)
    print("%-52s %8.4f %8.4f" % (q["label"][-50:], own, max(others)), flush=True)
mat, mis = np.array(mat), np.array(mis)
print("\nmatched   n=%-3d mean %.4f  sd %.4f" % (len(mat), mat.mean(), mat.std()))
print("mismatched n=%-3d mean %.4f  sd %.4f" % (len(mis), mis.mean(), mis.std()))
print("separation: matched - mismatched = %+.4f  (%.1fx)"
      % (mat.mean() - mis.mean(), mat.mean() / max(1e-6, abs(mis.mean()))))
won = sum(1 for i, q in enumerate(PAIRS)
          if mat[i] > max(cos(LV[k] - PMU, (q["h"] @ J.T) - AMU) for k in LV if k != (q["p"], q["k"])))
print("own label beats every other label: %d/%d positions" % (won, len(PAIRS)))
print("TEST_PROSE_LABEL_DONE")
