#!/usr/bin/env python3
"""
Evolve the TEMPLATE, not the phrase. Claude proposes, gradient search refines, scores go back.

Loop per round:
  1. Claude proposes N templates, given every prior template with its measured scores
  2. score each: |mod|, bare, disc, bonus   (held-out blogpost positions)
  3. GCG-refine the round's best: append K free tokens to the instruction and optimise them to
     raise the bare score, keeping the {x}/{y} structure intact
  4. feed the frontier plus whatever the gradient search found back to Claude

Measured facts handed to Claude, so it proposes from evidence rather than taste:
  * |mod| (the modulation magnitude) correlates +0.71 with discrimination and -0.87 with the
    wrapper bonus. Stronger modulation IS better and is what removes the hack.
  * emphatic-but-short wins: 'really focus' > 'focus' > 'concentrate' > 'think'. Capitalisation is
    irrelevant ('REALLY FOCUS' ties 'really focus').
  * elaboration HURTS: 'let X fill your whole mind and crowd out everything else' ranked last of 14.
  * y-first ordering wins: all top 8 of 107 screened were y-first.
  * binding {x} grammatically to the act of writing is worth +0.046 over isolating it.
  * long tails after {y} hurt (bare 0.218 -> 0.175, bonus +0.120 -> +0.226).
"""
import argparse, json, os, re, sys, time
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--propose", type=int, default=10)
ap.add_argument("--gcg-steps", type=int, default=24)
ap.add_argument("--gcg-slot", type=int, default=6)
ap.add_argument("--carriers", type=int, default=2)
ap.add_argument("--lo", type=int, default=76)
ap.add_argument("--hi", type=int, default=96)
ap.add_argument("--llm", default="claude-sonnet-5")
ap.add_argument("--out", default="/workspace/inv/results/template_evolve.json")
A = ap.parse_args()
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
W_E = model.get_input_embeddings().weight
CARS = C.CARRIERS_RECOVERED[: A.carriers]

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
keys = list(g350)[A.lo:A.hi]
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
FILL = [" ".join(rr.choice(words) for _ in range(rr.randint(3, 9))) for _ in range(24)]
ALL = sorted(set(CORE.values()) | set(FULL.values()) | set(FILL))
print("[e] %d held-out positions | %d strings | %d carriers" % (len(keys), len(ALL), len(CARS)),
      flush=True)


def measure(tpl):
    if "{x}" not in tpl or "{y}" not in tpl:
        return None
    try:
        G = C.Grid(tok, [tpl], CARS, 42, J, dev)
    except Exception:
        return None
    mods, bm, bx, pm = [], [], [], []
    with torch.no_grad():
        for ci in range(len(CARS)):
            rv = G.read(model, ALL, L42, carrier=ci, max_tok=32)
            pmu = torch.stack([rv[f] for f in FILL]).mean(0)
            for k in keys:
                dv = rv[CORE[k]] - pmu
                mods.append(float(dv.norm()))
                u = dv / dv.norm().clamp(min=1e-8)
                bm.append(float(u @ TGT[k]))
                uf = (rv[FULL[k]] - pmu) / (rv[FULL[k]] - pmu).norm().clamp(min=1e-8)
                pm.append(float(uf @ TGT[k]))
                bx += [float(u @ TGT[k2]) for k2 in keys[:6] if k2 != k]
    r = {"mod": float(np.mean(mods)), "bare": float(np.mean(bm)),
         "mis": float(np.mean(bx)), "pizza": float(np.mean(pm))}
    r["disc"] = r["bare"] - r["mis"]
    r["bonus"] = r["pizza"] - r["bare"]
    r["score"] = r["disc"] - r["bonus"]
    return r


def gcg_refine(tpl, steps, slot):
    """append `slot` free tokens to the template and optimise them to raise the bare score."""
    pat = re.compile(r"^ ?[A-Za-z][A-Za-z'-]{1,}$")
    ok = [t for t, s in enumerate(tok.batch_decode([[i] for i in range(W_E.shape[0])]))
          if s and s.isascii() and pat.match(s)]
    allow = torch.tensor(ok, device=dev)
    cur = allow[torch.randint(0, len(allow), (slot,), device=dev)]
    G = C.Grid(tok, [tpl + " %s"], CARS, 42, J, dev)   # placeholder, rebuilt per step below
    best = (measure(tpl)["bare"], "")
    for it in range(steps):
        txt = tok.decode(cur)
        r = measure(tpl + " " + txt)
        if r and r["bare"] > best[0]:
            best = (r["bare"], txt)
        # single random substitution, hill-climb (cheap stand-in for full GCG on a template)
        j = int(torch.randint(0, slot, (1,)))
        cand = cur.clone()
        cand[j] = allow[int(torch.randint(0, len(allow), (1,)))]
        r2 = measure(tpl + " " + tok.decode(cand))
        if r2 and r2["bare"] >= (r["bare"] if r else -9):
            cur = cand
    return best


SYS = """You are designing a PROMPT TEMPLATE for a mechanistic-interpretability probe.

The template has two slots. {x} is a short phrase. {y} is a fixed neutral carrier sentence. The
template is rendered as a user turn; the carrier is then PREFILLED as the assistant's reply, and we
read the model's layer-42 residual stream at the carrier's token positions. The goal: that read
should depend as much as possible on WHAT THE MODEL IS HOLDING IN MIND ({x}), and as little as
possible on anything else.

You are optimising three measured quantities:
  |mod|   magnitude of the read minus the mean read over random {x} fillers. HIGHER is better.
  disc    how much better a correct {x} scores than an {x} taken from a different target. HIGHER.
  bonus   how much a padded/boilerplate {x} outscores a plain one. LOWER is better (0 is ideal).

What measurement has established, over 130+ templates:
  * |mod| correlates +0.71 with disc and -0.87 with bonus. Stronger modulation is the single best
    predictor of a good template. Maximise it.
  * Emphatic but SHORT wins: "really focus on" > "focus on" > "concentrate on" > "think about".
    Capitalisation is irrelevant -- "REALLY FOCUS" ties "really focus".
  * Elaboration HURTS. "Let {x} fill your whole mind and crowd out everything else" ranked LAST of
    14 intensity variants (lowest |mod|, highest bonus).
  * {y}-FIRST ordering wins: the top 8 of 107 screened templates all state the carrier first, then
    the thought.
  * Bind {x} grammatically to the ACT OF WRITING ("as you write it, focus on {x}") rather than
    isolating it in its own sentence. Worth +0.046.
  * Keep whatever follows {y} SHORT. Long trailing instructions lower |mod| and raise bonus.
  * Demanding verbatim reproduction of {y} keeps the carrier near-certain, which is wanted: if the
    carrier were a real choice the model would spend capacity choosing instead of holding {x}.

Current best: 'Repeat this line verbatim: "{y}"\\n\\nAs you write it, focus on the following idea:
"{x}". Output only the sentence.'  (bare 0.358, disc 0.254, bonus +0.027)

Propose templates that beat it. Both {x} and {y} must appear exactly once. Vary real structure, not
synonyms. Reply with ONE template per line, using \\n for newlines, and nothing else."""

hist = []
for seed in ['Repeat this line verbatim: "{y}"\n\nAs you write it, focus on the following idea: '
             '"{x}". Output only the sentence.',
             C.TEMPLATES_RECOVERED[0],
             'Repeat this line verbatim: "{y}"\n\nAs you write it, really focus on "{x}". '
             'Output only the sentence.']:
    r = measure(seed)
    if r:
        hist.append({"tpl": seed, **r})
        print("  seed  score %.4f  |mod| %.2f  bare %.4f  bonus %+.4f  %r"
              % (r["score"], r["mod"], r["bare"], r["bonus"], seed[:58]), flush=True)

key = os.environ.get("ANTHROPIC_API_KEY", "")
cl = None
if key:
    import anthropic
    cl = anthropic.Anthropic(api_key=key)
for rd in range(1, A.rounds + 1):
    if cl is None:
        break
    top = sorted(hist, key=lambda h: -h["score"])[:12]
    tbl = "\n".join("  score %.4f | mod %.2f | bare %.4f | disc %.4f | bonus %+.4f | %s"
                    % (h["score"], h["mod"], h["bare"], h["disc"], h["bonus"],
                       h["tpl"].replace("\n", "\\n")) for h in top)
    msg = ("Templates measured so far, best first:\n%s\n\nPropose %d new templates."
           % (tbl, A.propose))
    txt = ""
    for att in range(4):
        try:
            r = cl.messages.create(model=A.llm, max_tokens=2000,
                                   system=[{"type": "text", "text": SYS,
                                            "cache_control": {"type": "ephemeral"}}],
                                   messages=[{"role": "user", "content": msg}])
            txt = "".join(getattr(b, "text", "") for b in r.content
                          if getattr(b, "type", "") == "text")
            break
        except Exception as e:
            print("  [llm] %s (retry %d)" % (type(e).__name__, att), flush=True)
            time.sleep(3 * (att + 1))
    new = []
    for ln in txt.splitlines():
        t = ln.strip().strip('"').replace("\\n", "\n")
        if "{x}" in t and "{y}" in t and t.count("{x}") == 1 and t.count("{y}") == 1:
            new.append(t)
    print("\n=== round %d: %d proposals ===" % (rd, len(new)), flush=True)
    for t in new[: A.propose]:
        r = measure(t)
        if r:
            hist.append({"tpl": t, **r})
            print("  score %.4f  |mod| %.2f  bare %.4f  disc %.4f  bonus %+.4f  %r"
                  % (r["score"], r["mod"], r["bare"], r["disc"], r["bonus"],
                     t.replace("\n", "\\n")[:70]), flush=True)
    best = max(hist, key=lambda h: h["score"])
    if A.gcg_steps:
        b, suf = gcg_refine(best["tpl"], A.gcg_steps, A.gcg_slot)
        if suf and b > best["bare"]:
            t2 = best["tpl"] + " " + suf
            r2 = measure(t2)
            if r2:
                hist.append({"tpl": t2, **r2})
                print("  [gcg] appended %r -> bare %.4f (was %.4f)"
                      % (suf, r2["bare"], best["bare"]), flush=True)

hist.sort(key=lambda h: -h["score"])
print("\n=== final top 8 ===")
for h in hist[:8]:
    print("  score %.4f | mod %.2f | bare %.4f | disc %.4f | bonus %+.4f\n      %r"
          % (h["score"], h["mod"], h["bare"], h["disc"], h["bonus"], h["tpl"]))
json.dump(hist, open(A.out, "w"), indent=1)
print("\nEVOLVE_DONE", flush=True)
