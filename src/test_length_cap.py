#!/usr/bin/env python3
"""
Is the 16-token budget what stalled the plain arm?

Both arms' mean rollout length climbed to ~15.1 of a hard cap of 16 and pinned there exactly as
reward stopped gaining (~it75), and both adopted slash-synonym-lists -- which is what a
length-starved policy does: `eccentric/hippie/alternative` is three concepts in one phrase's budget.

Two different things could bind, and they need separating:
  HARD CAP    -- generate() stops at max_new_tokens.
  PROMPT TEXT -- the prompt itself says "At most 16 tokens", a limit the policy was trained under.
So three conditions:
  A  prompt says 16, cap 16   -- the trained condition, the baseline
  B  prompt says 16, cap 48   -- does it WANT more room, unprompted? If it still stops near 16, the
                                prompt is what binds and raising the cap alone buys nothing.
  C  prompt says B,  cap B    -- matched, the best case for a longer budget
Reports mean tokens emitted, the fraction ending naturally at EOS (rather than being truncated), and
mean/best-of-N reward.

GOTCHA this script exists to avoid: Grid.read_all truncates candidate strings to `max_tok` tokens
(training passes max_tok=A.max_new=16). Reading a 48-token candidate at max_tok=16 would silently
score only its first 16 tokens and show a fake null result. max_tok is therefore tied to the budget.
"""
import argparse, json, os, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--probe-npy", default="/workspace/inv/data/holdout_blogpost.npy")
p.add_argument("--pmu", default="/workspace/inv/ckpts/rl_v2_plain/pmu_db4a6b8ee6.npy")
p.add_argument("--budgets", default="16,24,32,48")
p.add_argument("--group", type=int, default=4, help="samples per activation at temp 1.0")
p.add_argument("--whiten", type=int, default=0)
p.add_argument("--whitener", default="/workspace/inv/data/meansub/natural_whitener_jspace.npz")
p.add_argument("--out", default="/workspace/inv/results/length_cap.json")
A = p.parse_args()
dev = "cuda"
EOS = 248046                                    # <|im_end|> -- verified, not guessed

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = model.model
INJ, LEFT, RIGHT = C.marker_ids(tok)
HOOK, L42 = {"vec": None, "ids": None}, {}


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


def build_prompt(stated):
    JOB = ("You are shown an internal activation vector captured from a language model at a single "
           "position while it was reading some text. The vector is enclosed in <concept> tags.\n\n"
           "<concept>%s</concept>\n\n"
           "Your job: write the short phrase that this state is holding in mind.\n\n"
           "How it is judged. Your phrase is placed into a prompt of the form\n"
           '  Focus on the following idea: "<your phrase>" while writing the following phrase: '
           '"<a fixed unrelated sentence>"\n'
           "The model then writes that fixed sentence, and we read its internal state while it "
           "does so. You score well when that state matches the state you were given.\n\n"
           "So write what the model should be THINKING ABOUT -- not a description of a vector, and "
           "not a comment on the task. Natural, fluent English. At most %d tokens. Output only the "
           "phrase." % (C.INJ_CHAR, stated))
    txt = tok.apply_chat_template([{"role": "user", "content": JOB}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    ids = torch.tensor(tok.encode(txt, add_special_tokens=False), device=dev)
    at = (ids == INJ).nonzero().flatten()
    assert at.numel() == 1, "prompt needs exactly one marker"
    k = int(at[0])
    assert int(ids[k - 1]) == LEFT and int(ids[k + 1]) == RIGHT, "marker neighbours wrong"
    return ids


J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, 42, J, dev)
assert GRID.sig()[:10] == "db4a6b8ee6", "grid drifted: %s" % GRID.sig()[:10]
PMU = torch.from_numpy(np.load(A.pmu)).to(dev).float()
Wm = torch.from_numpy(np.load(A.whitener)["W_ridge0.1"]).to(dev).float()
V = []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=4096, columns=["activation_vector"]):
    V.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in V) >= 20000:
        break
AMU = torch.from_numpy(np.concatenate(V)[:20000]).mean(0).to(dev) @ J.T
ACT = torch.from_numpy(np.load(A.probe_npy).astype("float32"))
N = ACT.shape[0]
TGT = torch.stack([(ACT[i].to(dev) @ J.T) - AMU for i in range(N)])
print("[l] grid %s | %d probes | |PMU| %.2f" % (GRID.sig()[:10], N, PMU.norm()), flush=True)

model = PeftModel.from_pretrained(model, A.adapter)
model.eval()


@torch.no_grad()
def gen(PIDS, cap, G):
    """-> list per probe of (text, n_tokens, ended_naturally)"""
    PLEN = PIDS.numel()
    out = [[] for _ in range(N)]
    per = max(1, 64 // G)
    for s in range(0, N, per):
        sub = ACT[s:s + per].to(dev).float()
        B = sub.shape[0] * G
        HOOK["vec"] = sub.repeat_interleave(G, 0)
        try:
            g = model.generate(input_ids=PIDS.unsqueeze(0).expand(B, -1).contiguous(),
                               attention_mask=torch.ones(B, PLEN, device=dev, dtype=torch.long),
                               max_new_tokens=cap, do_sample=True, temperature=1.0,
                               top_p=1.0, top_k=0, pad_token_id=EOS)
        finally:
            HOOK["vec"] = None
        new = g[:, PLEN:]
        for j in range(new.shape[0]):
            row = new[j].tolist()
            nat = EOS in row
            ntok = row.index(EOS) if nat else len(row)
            txt = tok.decode(row[:ntok], skip_special_tokens=True).strip() or " the"
            out[s + j // G].append((txt, ntok, nat))
    return out


def score(rolls, max_tok):
    uniq = sorted({t for r in rolls for (t, _, _) in r})
    with model.disable_adapter(), torch.no_grad():
        RV = GRID.read_all(model, uniq, L42, max_tok=max_tok)

    def pv(s):
        v = RV[s] - PMU
        return v @ Wm.T if A.whiten else v

    def tv(i):
        return TGT[i] @ Wm.T if A.whiten else TGT[i]

    mean_, best_ = [], []
    for i, r in enumerate(rolls):
        cs = [float((pv(t) @ tv(i)) / (pv(t).norm() * tv(i).norm() + 1e-8)) for (t, _, _) in r]
        mean_.append(float(np.mean(cs))); best_.append(max(cs))
    return float(np.mean(mean_)), float(np.mean(best_))


BUDG = [int(x) for x in A.budgets.split(",")]
CONDS = [("A prompt16/cap16", 16, 16)]
CONDS += [("B prompt16/cap%d" % b, 16, b) for b in BUDG if b != 16]
CONDS += [("C prompt%d/cap%d" % (b, b), b, b) for b in BUDG if b != 16]
RES = {}
print("\n%-20s %7s %8s %9s %9s %9s" % ("condition", "cap", "mean_tok", "%natural", "reward", "best4"))
for name, stated, cap in CONDS:
    PIDS = build_prompt(stated)
    rolls = gen(PIDS, cap, A.group)
    toks = [n for r in rolls for (_, n, _) in r]
    nat = [x for r in rolls for (_, _, x) in r]
    # Reward is computed on the RE-TOKENIZED decoded text, not on the generated ids. If decoding +
    # re-tokenizing lengthens a string past the read's max_tok, its tail is silently clipped and the
    # policy is rewarded for something shorter than it wrote. Measure how often that happens.
    retok = [len(tok(t, add_special_tokens=False).input_ids) for r in rolls for (t, _, _) in r]
    gen_n = [n for r in rolls for (_, n, _) in r]
    over16 = 100.0 * float(np.mean([x > 16 for x in retok]))
    drift = float(np.mean([a - b for a, b in zip(retok, gen_n)]))
    # read at >= the cap so a long candidate is not silently truncated by the grid
    mr, br = score(rolls, max_tok=max(cap + 4, 20))
    RES[name] = {"stated": stated, "cap": cap, "mean_tok": float(np.mean(toks)),
                 "pct_natural": 100.0 * float(np.mean(nat)), "reward": mr, "best4": br,
                 "pct_retok_over16": over16, "retok_minus_gen": drift,
                 "examples": [r[0][0] for r in rolls[:5]]}
    print("%-20s %7d %8.1f %8.0f%% %9.4f %9.4f"
          % (name, cap, np.mean(toks), 100 * np.mean(nat), mr, br), flush=True)
    print("      retok-gen %+.2f tok | %.0f%% would exceed the training read's max_tok=16"
          % (drift, over16))
    for e in RES[name]["examples"][:3]:
        print("      %r" % e)

base = RES["A prompt16/cap16"]
print("\n=== vs the trained condition (A) ===")
for k, r in RES.items():
    if k == "A prompt16/cap16":
        continue
    print("  %-20s tok %+6.1f  reward %+.4f  %s"
          % (k, r["mean_tok"] - base["mean_tok"], r["reward"] - base["reward"],
             "USES the room" if r["mean_tok"] > base["mean_tok"] + 1.5 else "stops anyway"))
print("\n  If B stops near 16 anyway, the PROMPT TEXT binds and raising the cap alone buys nothing.")
print("  If B runs long and scores higher, the HARD CAP was the limit.")
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(RES, open(A.out, "w"), indent=1)
print("\nwrote %s\nLENGTH_CAP_DONE" % A.out, flush=True)
