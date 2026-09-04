#!/usr/bin/env python3
"""
Interactive-ish tool: invert the hidden state at any position of any text, then refine.

Pipeline per target:
  1. the trained INVERTER proposes k candidates in one forward pass each (no search)
  2. each is scored in the carrier geometry -- the reward the inverter was trained on
  3. Sonnet sees the scored candidates and writes rewrites (reflective mutation)
  4. score, keep a frontier, repeat

Why the inverter as proposer rather than dictionary atoms: it is a learned one-shot map from state to
phrase, so round 0 is already near the useful region instead of starting from a bag of atoms. The
step-50 RL checkpoint is the recommended one -- later checkpoints score higher but collapse onto a
fixed prefix ("not pizza related wording but related ...") that is identical for every input.

  python3 evolve_inverter.py --text "your sentence here" --pos -1
  python3 evolve_inverter.py --blogpost --para 12          # a blog-post paragraph, last mark
  python3 evolve_inverter.py --text "..." --pos 7 --rounds 4 --propose 16
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inv_core as C

ap = argparse.ArgumentParser()
ap.add_argument("--text", default="")
ap.add_argument("--pos", type=int, default=-1, help="token index in --text; -1 = last")
ap.add_argument("--blogpost", action="store_true", help="use data/blogpost.txt")
ap.add_argument("--para", type=int, default=0, help="with --blogpost: which paragraph")
ap.add_argument("--policy", default="/workspace/inv/ckpts/rl/iter_000050")
ap.add_argument("--rounds", type=int, default=3, help="0 = inverter only, no LLM refinement")
ap.add_argument("--propose", type=int, default=12, help="inverter samples, and LLM rewrites/round")
ap.add_argument("--keep", type=int, default=10)
ap.add_argument("--carriers", type=int, default=3)
ap.add_argument("--layer", type=int, default=42)
ap.add_argument("--max-new", type=int, default=16)
ap.add_argument("--temp", type=float, default=1.1)
ap.add_argument("--whitener", default="/workspace/inv/data/meansub/natural_whitener_jspace.npz")
ap.add_argument("--acts-pool", default="/workspace/inv/data/prose_L42.parquet")
ap.add_argument("--llm", default="claude-sonnet-5")
ap.add_argument("--out", default="")
A = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
INJ, LEFT, RIGHT = C.marker_ids(tok)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
J = C.load_jlens(A.layer, dev)
MU, Wm = C.load_whitener(A.whitener, "0.1", dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[: A.carriers], A.layer, J, dev)
L42 = {}
base.model.layers[A.layer].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))

import pyarrow.parquet as pq
acc = []
for b in pq.ParquetFile(A.acts_pool).iter_batches(batch_size=4096, columns=["activation_vector"]):
    acc.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in acc) >= 20000:
        break
AMU = torch.from_numpy(np.concatenate(acc)[:20000]).mean(0).to(dev) @ J.T
PMU_F = os.path.join(os.path.dirname(A.policy.rstrip("/")), "pmu.npy")
PMU = (torch.tensor(np.load(PMU_F), device=dev) if os.path.exists(PMU_F)
       else GRID.prompt_mean(base, L42, n=64))
print("[t] |PMU| %.2f  |AMU| %.2f  grid %dx%d" % (float(PMU.norm()), float(AMU.norm()),
      GRID.n_tpl, GRID.n_car), flush=True)

# ---- the target ----
CPRE, CPOST = C.chat_wrap_ids(tok)
if A.blogpost:
    paras = [x.strip() for x in open("/workspace/inv/data/blogpost.txt").read().split("\n\n")
             if x.strip()]
    TEXT = paras[A.para % len(paras)]
else:
    TEXT = A.text or "do you want to make it or do you just want to be held?"
ids = tok(TEXT, add_special_tokens=False).input_ids[:256]
with torch.no_grad():
    base(input_ids=torch.tensor([CPRE + ids + CPOST], device=dev))
    H = L42["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
P = A.pos if A.pos >= 0 else len(ids) - 1
P = max(0, min(P, len(ids) - 1))
HVEC = H[P].clone()
CTX = tok.decode(ids[max(0, P - 30):P + 1])
print("\n[t] text  : %r" % TEXT[:110])
print("[t] target: token %d = %r" % (P, tok.decode([ids[P]])))
print("[t] context ...%r\n" % CTX[-70:], flush=True)

TW = ((HVEC @ J.T) - AMU) @ Wm.T
TW = TW / TW.norm().clamp(min=1e-8)


PEFT = [None]


def score(strings):
    with torch.no_grad():
        if PEFT[0] is not None:
            with PEFT[0].disable_adapter():
                v = GRID.read(base, strings, L42, carrier=0, max_tok=A.max_new)
        else:
            v = GRID.read(base, strings, L42, carrier=0, max_tok=A.max_new)
    out = {}
    for s in strings:
        a = (v[s] - PMU) @ Wm.T
        out[s] = float((a @ TW) / a.norm().clamp(min=1e-8))
    return out


# ---- round 0: the inverter proposes ----
JOB = ("You are shown an internal activation vector captured from a language model at a single "
       "position while it was reading some text. The vector is enclosed in <concept> tags.\n\n"
       "<concept>%s</concept>\n\n"
       "Your job: write the short phrase that this state is holding in mind.\n\n"
       "How it is judged. Your phrase is placed into a prompt of the form\n"
       '  Focus on the following idea: "<your phrase>" while writing the following phrase: '
       '"<a fixed unrelated sentence>"\n'
       "The model then writes that fixed sentence, and we read its internal state while it does "
       "so. You score well when that state matches the state you were given.\n\n"
       "So write what the model should be THINKING ABOUT -- not a description of a vector, and not "
       "a comment on the task. Natural, fluent English. At most %d tokens. Output only the phrase."
       % (C.INJ_CHAR, A.max_new))
PIDS = torch.tensor(tok.encode(tok.apply_chat_template(
    [{"role": "user", "content": JOB}], tokenize=False, add_generation_prompt=True,
    enable_thinking=False), add_special_tokens=False), device=dev)
PLEN = PIDS.numel()

m = PeftModel.from_pretrained(base, A.policy, adapter_name="inv")
m.set_adapter("inv")
inner = m.base_model.model.model
HK = {"vec": None, "ids": None}
inner.register_forward_pre_hook(
    lambda mod, a, kw: HK.__setitem__("ids", kw.get("input_ids", a[0] if a else None)),
    with_kwargs=True)


def _inj(mod, a, o):
    r = o[0] if isinstance(o, tuple) else o
    if HK["vec"] is None or HK["ids"] is None:
        return o
    if tuple(HK["ids"].shape) != tuple(r.shape[:-1]) or not bool((HK["ids"] == INJ).any()):
        return o
    n = C.inject_at_marker(HK["ids"], r, HK["vec"], INJ, LEFT, RIGHT)
    return (n,) + tuple(o[1:]) if isinstance(o, tuple) else n


inner.layers[1].register_forward_hook(_inj)
with torch.no_grad():
    HK["vec"] = HVEC.unsqueeze(0).expand(A.propose, -1).contiguous().float()
    g = m.generate(input_ids=PIDS.unsqueeze(0).expand(A.propose, -1).contiguous(),
                   attention_mask=torch.ones(A.propose, PLEN, device=dev, dtype=torch.long),
                   max_new_tokens=A.max_new, do_sample=True, temperature=A.temp, top_p=1.0,
                   top_k=0, pad_token_id=tok.eos_token_id)
    HK["vec"] = None
cands = sorted({t.strip() for t in tok.batch_decode(g[:, PLEN:], skip_special_tokens=True)
                if t.strip()})
PEFT[0] = m  # from here, score() runs with the adapter disabled
pop = sorted(score(cands).items(), key=lambda kv: -kv[1])
print("=== round 0: the inverter, one forward pass each ===")
for s, v in pop[:A.keep]:
    print("  %+.4f  %r" % (v, s[:78]))

# ---- LLM refinement ----
SYS = """You are refining a phrase that describes what a language model was holding in mind at one
position in a text.

How scoring works: the phrase is inserted into "Focus on the following idea: <PHRASE> while writing
the following phrase: <a fixed unrelated sentence>". The model then writes that fixed sentence and we
read its internal state. A phrase scores well when that state matches the target state.

You are given the surrounding text, and every candidate tried so far with its score. Some candidates
come from a small model trained to do this in one shot; your job is to do better than it.

Guidance that comes from measurement, not taste:
- Naming the mental POSTURE beats restating the words. "not searching anymore but found it" beat
  paraphrases of "There it is!". Contrastive form -- saying what the state is NOT to pin what it is --
  works well.
- Inferring the implicature beats copying. For "could go to London or Paris for a week!" the best
  answer was "opportunity cost".
- Do NOT pad with boilerplate. An RL run discovered that prefixing everything with "not pizza related
  wording but related ..." scores highly and says nothing. That is the failure mode to avoid: if a
  phrase would score the same for a completely different text, it is worthless.
- 4-16 words, natural English, no meta-commentary about vectors or tasks.

Reply with ONE candidate per line, nothing else. No numbering or quotes."""

if A.rounds > 0:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("\n[t] ANTHROPIC_API_KEY unset -- stopping after the inverter's proposals", flush=True)
    else:
        import anthropic
        cl = anthropic.Anthropic(api_key=key)
        for rd in range(1, A.rounds + 1):
            hist = "\n".join("  %+.4f  %s" % (v, s) for s, v in pop[:18])
            msg = ("Text, ending at the target position:\n  ...%s\n\nThe target token is %r.\n\n"
                   "Candidates so far (score, phrase):\n%s\n\nWrite %d new candidates."
                   % (CTX[-220:], tok.decode([ids[P]]), hist, A.propose))
            txt = ""
            for att in range(4):
                try:
                    r = cl.messages.create(model=A.llm, max_tokens=1600,
                                           system=[{"type": "text", "text": SYS,
                                                    "cache_control": {"type": "ephemeral"}}],
                                           messages=[{"role": "user", "content": msg}])
                    txt = "".join(getattr(b, "text", "") for b in r.content
                                  if getattr(b, "type", "") == "text")
                    break
                except Exception as e:
                    print("  [llm] %s (retry %d)" % (type(e).__name__, att), flush=True)
                    time.sleep(3 * (att + 1))
            new = [re.sub(r'^[\s\-\*\d\.\)"]+|"+$', "", x).strip() for x in txt.splitlines()]
            new = [x for x in new if 3 <= len(x.split()) <= 28][:A.propose]
            if not new:
                print("=== round %d: no proposals ===" % rd, flush=True)
                continue
            sc = score(new)
            pop = sorted({**dict(pop), **sc}.items(), key=lambda kv: -kv[1])[:max(A.keep, 24)]
            print("\n=== round %d: %d LLM rewrites, best now %+.4f ===" % (rd, len(new), pop[0][1]))
            for s, v in pop[:A.keep]:
                print("  %+.4f  %r" % (v, s[:78]))

print("\n=== final frontier ===")
for s, v in pop[:A.keep]:
    print("  %+.4f  %r" % (v, s[:82]))
if A.out:
    json.dump({"text": TEXT, "pos": P, "token": tok.decode([ids[P]]), "ctx": CTX,
               "policy": A.policy, "frontier": [{"phrase": s, "score": v} for s, v in pop]},
              open(A.out, "w"), indent=1)
    print("\nwrote %s" % A.out)
print("EVOLVE_DONE", flush=True)
