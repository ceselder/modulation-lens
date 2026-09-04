#!/usr/bin/env python3
"""
Are the slash-lists and form-descriptions the RL policy emits faithful readouts, or generic filler?

By step 60 both RL arms emit strings like 'City with eccentric/hippie/alternative people' and
'Transcript of spoken dialogue (interview/conversation)'. Two competing readings:

  HACK      -- a slash list of near-synonyms sprays the target subspace, and "this is a transcript"
               is true of most of the document, so both buy score on ANY target.
  FAITHFUL  -- the target is a MEAN-POOLED activation, hence genuinely a mixture of concepts, so a
               list may sit closer to it than any single term; and the form-descriptions land on
               positions that really are inside a speaker-labelled transcript.

These make opposite predictions about MISMATCHED targets, which is what this measures:
  bare = cos(emitted_k, target_k);  mis = mean_{k2!=k} cos(emitted_k, target_k2);  disc = bare - mis
Generic filler -> mis high, disc ~0. Faithful -> mis low, disc large.

For form-descriptions there is a third possibility: correctly reading a property SHARED by all
dialogue positions. That looks like a hack under a blanket `mis` but is not. So `mis` is split into
mis_same (mismatched targets of the same position type, dialogue vs not) and mis_diff (the other
type). Faithful-but-coarse => mis_same high, mis_diff low. Generic => both high.

Reported per adapter and per string category, under BOTH plain and whitened cosine, because the two
arms were trained on different metrics and each should be judged on its own as well as on the shared
one. Generation is greedy so the string is the policy's modal output, not a sample.
"""
import argparse, json, os, re, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

p = argparse.ArgumentParser()
p.add_argument("--adapters", required=True,
               help="comma list of label=path, e.g. sft=/workspace/inv/ckpts/sft/final")
p.add_argument("--probe-npy", default="/workspace/inv/data/holdout_blogpost.npy")
p.add_argument("--probe-meta", default="/workspace/inv/data/holdout_blogpost.jsonl")
p.add_argument("--pmu", default="/workspace/inv/ckpts/rl_v2_plain/pmu_db4a6b8ee6.npy",
               help="the EXACT PMU the runs trained against, not a re-estimate")
p.add_argument("--whitener", default="/workspace/inv/data/meansub/natural_whitener_jspace.npz")
p.add_argument("--max-new", type=int, default=16)
p.add_argument("--out", default="/workspace/inv/results/emitted_disc.json")
A = p.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = model.model
INJ, LEFT, RIGHT = C.marker_ids(tok)
HOOK = {"vec": None, "ids": None}
L42 = {}


def _stash(mod, a, kw):
    ids = kw.get("input_ids")
    if ids is None and a:
        ids = a[0]
    HOOK["ids"] = ids


def _inject(mod, a, out):
    resid = out[0] if isinstance(out, tuple) else out
    ids, vec = HOOK["ids"], HOOK["vec"]
    if vec is None or ids is None or tuple(ids.shape) != tuple(resid.shape[:-1]):
        return out
    if not bool((ids == INJ).any()):
        return out
    new = C.inject_at_marker(ids, resid, vec, INJ, LEFT, RIGHT)
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


inner.register_forward_pre_hook(_stash, with_kwargs=True)
inner.layers[1].register_forward_hook(_inject)
inner.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))

# --- prompt: byte-identical to inv_train.py's, or the policy is off-distribution ---
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
PROMPT_TXT = tok.apply_chat_template([{"role": "user", "content": JOB}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
PIDS = torch.tensor(tok.encode(PROMPT_TXT, add_special_tokens=False), device=dev)
PLEN = PIDS.numel()
_at = (PIDS == INJ).nonzero().flatten()
assert _at.numel() == 1, "prompt needs exactly one marker, found %d" % _at.numel()
_k = int(_at[0])
assert int(PIDS[_k - 1]) == LEFT and int(PIDS[_k + 1]) == RIGHT, "marker neighbours wrong"
print("[p] prompt %d tok, marker %d, neighbours verified" % (PLEN, _k), flush=True)

J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, 42, J, dev)
assert GRID.sig()[:10] == "db4a6b8ee6", "grid drifted from the trained one: %s" % GRID.sig()[:10]
PMU = torch.from_numpy(np.load(A.pmu)).to(dev).float()
z = np.load(A.whitener)
Wm = torch.from_numpy(z["W_ridge0.1"]).to(dev).float()
V = []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=4096, columns=["activation_vector"]):
    V.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in V) >= 20000:
        break
AMU = torch.from_numpy(np.concatenate(V)[:20000]).mean(0).to(dev) @ J.T
print("[p] grid %s | |PMU| %.2f | |AMU| %.2f" % (GRID.sig()[:10], PMU.norm(), AMU.norm()), flush=True)

ACT = torch.from_numpy(np.load(A.probe_npy).astype("float32"))
META = [json.loads(l) for l in open(A.probe_meta)] if os.path.exists(A.probe_meta) else []
N = ACT.shape[0]
TGT = torch.stack([(ACT[i].to(dev) @ J.T) - AMU for i in range(N)])
# position type: is this inside a speaker-labelled transcript?
DLG = [bool(re.search(r"\b[A-Z]:\s", (META[i].get("ctx", "") if i < len(META) else "")))
       for i in range(N)]
print("[p] %d probe positions, %d dialogue-typed" % (N, sum(DLG)), flush=True)

FORM = re.compile(r"transcript|dialogu|conversation|interview|speech|spoken|spoke|speaker",
                  re.I)


@torch.no_grad()
def emit(m):
    """greedy, one string per probe activation"""
    out = []
    for s in range(0, N, 16):
        sub = ACT[s:s + 16].to(dev).float()
        B = sub.shape[0]
        HOOK["vec"] = sub
        try:
            g = m.generate(input_ids=PIDS.unsqueeze(0).expand(B, -1).contiguous(),
                           attention_mask=torch.ones(B, PLEN, device=dev, dtype=torch.long),
                           max_new_tokens=A.max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        finally:
            HOOK["vec"] = None
        out += [t.strip() or " the" for t in tok.batch_decode(g[:, PLEN:], skip_special_tokens=True)]
    return out


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def stats(strs, RV, whiten):
    """-> per-category bare/mis/disc, with mis split by same-vs-different position type"""
    def pv(s):
        v = RV[s] - PMU
        return v @ Wm.T if whiten else v

    def tv(i):
        return TGT[i] @ Wm.T if whiten else TGT[i]

    cats = {}
    for i, s in enumerate(strs):
        c = "SLASH" if "/" in s else ("FORM" if FORM.search(s) else "PLAIN")
        a = pv(s)
        bare = cos(a, tv(i))
        same = [cos(a, tv(j)) for j in range(N) if j != i and DLG[j] == DLG[i]]
        diff = [cos(a, tv(j)) for j in range(N) if j != i and DLG[j] != DLG[i]]
        for key in (c, "ALL"):
            d = cats.setdefault(key, {"n": 0, "bare": [], "same": [], "diff": []})
            d["n"] += 1
            d["bare"].append(bare)
            d["same"] += same
            d["diff"] += diff
    out = {}
    for k, d in cats.items():
        mis = float(np.mean(d["same"] + d["diff"])) if d["same"] + d["diff"] else 0.0
        out[k] = {"n": d["n"], "bare": float(np.mean(d["bare"])), "mis": mis,
                  "disc": float(np.mean(d["bare"])) - mis,
                  "mis_same": float(np.mean(d["same"])) if d["same"] else None,
                  "mis_diff": float(np.mean(d["diff"])) if d["diff"] else None}
    return out


RES = {}
cur = None
for spec in A.adapters.split(","):
    label, path = spec.split("=", 1)
    print("\n######## %s ########" % label, flush=True)
    m = model if path.lower() == "base" else PeftModel.from_pretrained(model, path)
    strs = emit(m)
    # The grid read must run on the BASE model with the adapter disabled, exactly as
    # score_strings() does during training -- the policy writes the phrase, the base model is what
    # gets modulated by it.
    uniq = sorted(set(strs))
    if hasattr(m, "disable_adapter"):
        with m.disable_adapter(), torch.no_grad():
            RV = GRID.read_all(m, uniq, L42, max_tok=32)
    else:
        with torch.no_grad():
            RV = GRID.read_all(m, uniq, L42, max_tok=32)
    RES[label] = {"strings": strs,
                  "plain": stats(strs, RV, False),
                  "whitened": stats(strs, RV, True)}
    nsl = sum("/" in s for s in strs); nfm = sum(bool(FORM.search(s)) and "/" not in s for s in strs)
    print("  %d/%d slash, %d form, %d plain" % (nsl, len(strs), nfm, len(strs) - nsl - nfm))
    for met in ("plain", "whitened"):
        print("  -- %s cosine --" % met)
        print("     %-6s %4s %8s %8s %8s | %9s %9s" %
              ("cat", "n", "bare", "mis", "disc", "mis_same", "mis_diff"))
        for c in ("ALL", "SLASH", "FORM", "PLAIN"):
            r = RES[label][met].get(c)
            if not r:
                continue
            print("     %-6s %4d %8.4f %8.4f %8.4f | %9s %9s" %
                  (c, r["n"], r["bare"], r["mis"], r["disc"],
                   "%.4f" % r["mis_same"] if r["mis_same"] is not None else "-",
                   "%.4f" % r["mis_diff"] if r["mis_diff"] is not None else "-"))
    for s in strs[:6]:
        print("     e.g. %r" % s)
    if path.lower() != "base":
        model = m.unload()   # restore base in place for the next adapter

os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(RES, open(A.out, "w"), indent=1)
print("\nwrote %s" % A.out)
print("EMITTED_DISC_DONE", flush=True)
