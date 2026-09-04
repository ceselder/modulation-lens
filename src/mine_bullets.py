#!/usr/bin/env python3
"""Stage 1 of the four-bullet lens: mine DIVERSE warm-start bullets by high-temperature sampling
plus non-orthogonal matching pursuit.

The four-bullet reward is a RELAXATION of the single-phrase one -- w=[lambda,0,0,0] is feasible, so
NNLS can never score below the best single bullet. The entire gain therefore comes from DIVERSITY:
four rephrasings and NNLS simply zeroes three of them. But the single-phrase policy has no reason to
produce diverse lines, so it cannot bootstrap itself.

So: sample N completions per activation at high temperature from the existing policy, then select 4
by greedy NOMP on the composition objective -- take the best single, then repeatedly add whichever
remaining candidate most improves the NNLS reconstruction. Selection optimises the exact quantity
RL will be rewarded on, and it picks complements rather than near-duplicates by construction.
Output is an SFT corpus of (activation -> 4 bullets).
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
p.add_argument("--policy", default="/workspace/inv/ckpts/rl_v2_plain/iter_000375")
p.add_argument("--data", default="/workspace/inv/data/prose_L42.parquet")
p.add_argument("--pmu", default="/workspace/inv/ckpts/rl_v2_plain/pmu_db4a6b8ee6.npy")
p.add_argument("--out", default="/workspace/inv/data/bullets4.jsonl")
p.add_argument("--n-acts", type=int, default=6000)
p.add_argument("--skip", type=int, default=0,
               help="skip the first N activations, so shards do not overlap")
p.add_argument("--samples", type=int, default=16, help="high-temp candidates per activation")
p.add_argument("--bullets", type=int, default=4)
p.add_argument("--temp", type=float, default=1.4, help="high, to get DIVERSE candidates")
p.add_argument("--max-new", type=int, default=16)
p.add_argument("--n-carriers", type=int, default=6)
A = p.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
model = PeftModel.from_pretrained(model, A.policy, adapter_name="default").eval()
inner = model.get_base_model().model
INJ, LEFT, RIGHT = C.marker_ids(tok)
HOOK, L42 = {"vec": None, "ids": None}, {}
inner.register_forward_pre_hook(
    lambda m, a, kw: HOOK.__setitem__("ids", kw.get("input_ids") if kw.get("input_ids") is not None
                                      else (a[0] if a else None)), with_kwargs=True)


def _inj(m, a, out):
    resid = out[0] if isinstance(out, tuple) else out
    ids, vec = HOOK["ids"], HOOK["vec"]
    if vec is None or ids is None or tuple(ids.shape) != tuple(resid.shape[:-1]):
        return out
    if not bool((ids == INJ).any()):
        return out
    new = C.inject_at_marker(ids, resid, vec, INJ, LEFT, RIGHT)
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


inner.layers[1].register_forward_hook(_inj)
inner.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))

JOB = ("You are shown an internal activation vector captured from a language model at a single "
       "position while it was reading some text. The vector is enclosed in <concept> tags.\n\n"
       "<concept>%s</concept>\n\nYour job: write the short phrase that this state is holding in "
       "mind.\n\nHow it is judged. Your phrase is placed into a prompt of the form\n"
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
assert int((PIDS == INJ).sum()) == 1
EOS = 248046

J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[: A.n_carriers], 42, J, dev)
PMU = torch.from_numpy(np.load(A.pmu)).to(dev).float()
V, LAB = [], []
for b in pq.ParquetFile(A.data).iter_batches(batch_size=4096,
                                             columns=["activation_vector", "label"]):
    d = b.to_pydict()
    V.append(np.asarray(d["activation_vector"], dtype="float32")); LAB += d["label"]
    if sum(len(x) for x in V) >= A.n_acts:
        break
ACT = torch.from_numpy(np.concatenate(V)[A.skip: A.n_acts])
LAB = LAB[A.skip: A.n_acts]
AMU = ACT.mean(0).to(dev) @ J.T
print("[mine] %d activations | %d samples each at T=%.2f | select %d by NOMP"
      % (ACT.shape[0], A.samples, A.temp, A.bullets), flush=True)


def nnls_small(B, t):
    n = B.shape[0]
    bw, br = None, float("inf")
    for mask in range(1, 1 << n):
        idx = [k for k in range(n) if (mask >> k) & 1]
        S = B[idx].T
        sol = torch.linalg.lstsq(S, t.unsqueeze(1)).solution.squeeze(1)
        if bool((sol < -1e-8).any()):
            continue
        r = float((t - S @ sol).norm())
        if r < br:
            br = r
            w = torch.zeros(n, device=B.device, dtype=B.dtype)
            w[torch.tensor(idx, device=B.device)] = sol
            bw = w
    return bw if bw is not None else torch.zeros(n, device=B.device, dtype=B.dtype)


def compose_cos(B, t):
    w = nnls_small(B, t)
    r = w @ B
    return float((r @ t) / r.norm()) if float(r.norm()) > 1e-8 else 0.0


out = open(A.out, "w")
kept = 0
with torch.no_grad():
    for s in range(0, ACT.shape[0], 8):
        sub = ACT[s:s + 8].to(dev).float()
        B = sub.shape[0] * A.samples
        HOOK["vec"] = sub.repeat_interleave(A.samples, 0)
        try:
            g = model.generate(input_ids=PIDS.unsqueeze(0).expand(B, -1).contiguous(),
                               attention_mask=torch.ones(B, PLEN, device=dev, dtype=torch.long),
                               do_sample=True, temperature=A.temp, top_p=1.0, top_k=0,
                               max_new_tokens=A.max_new, pad_token_id=EOS)
        finally:
            HOOK["vec"] = None
        rows = g[:, PLEN:].tolist()
        for a in range(sub.shape[0]):
            cands = []
            for r in rows[a * A.samples:(a + 1) * A.samples]:
                cut = r.index(EOS) if EOS in r else len(r)
                txt = tok.decode(r[:cut], skip_special_tokens=True).strip()
                if txt and txt not in cands:
                    cands.append(txt)
            if len(cands) < 2:
                continue
            with model.disable_adapter():
                cv = GRID.read_all(model, cands, L42, max_tok=A.max_new)
            M = torch.stack([cv[c] - PMU for c in cands])
            t = (sub[a] @ J.T) - AMU
            t = t / t.norm().clamp(min=1e-8)
            # greedy NOMP: start from the best single, then add whichever candidate most improves
            # the NNLS composition. Picks complements, not near-duplicates.
            chosen = [int(max(range(len(cands)), key=lambda k: float(M[k] @ t / M[k].norm())))]
            base = compose_cos(M[chosen], t)
            while len(chosen) < A.bullets:
                best, bs = None, base
                for k in range(len(cands)):
                    if k in chosen:
                        continue
                    c = compose_cos(M[chosen + [k]], t)
                    if c > bs + 1e-5:
                        best, bs = k, c
                if best is None:
                    break
                chosen.append(best); base = bs
            out.write(json.dumps({"i": A.skip + s + a, "label": LAB[s + a],
                                  "bullets": [cands[k] for k in chosen],
                                  "compose_cos": round(base, 4),
                                  "single_cos": round(float(M[chosen[0]] @ t
                                                            / M[chosen[0]].norm()), 4)}) + "\n")
            kept += 1
        if s % 200 == 0:
            out.flush()
            print("[mine] %d/%d acts, %d kept" % (s, ACT.shape[0], kept), flush=True)
out.close()
print("[mine] wrote %d rows -> %s" % (kept, A.out), flush=True)
print("MINE_DONE", flush=True)
