#!/usr/bin/env python3
"""Does the published four-bullet checkpoint emit language-mentions on the TRAINING distribution,
or only on the blogpost probe? The probe is one document, so a pattern seen only there could be a
property of that text rather than of the policy."""
import sys, re, json
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C
dev, EOS, N = "cuda", 248046, 48
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = base.model
INJ, LEFT, RIGHT = C.marker_ids(tok)
HOOK = {"vec": None, "ids": None}
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
    new = C.inject_at_marker(ids, resid, vec, INJ, LEFT, RIGHT, "replace")
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


inner.layers[1].register_forward_hook(_inj)
model = PeftModel.from_pretrained(base, "/workspace/inv/ckpts/rl_b4_long4/iter_000200").eval()

# the EXACT 229-token prompt that checkpoint was trained with (pre-fix)
JOB = ("You are shown an internal activation vector captured from a language model at a single "
       "position while it was reading some text. The vector is enclosed in <concept> tags.\n\n"
       "<concept>%s</concept>\n\n"
       "Your job: write 4 short phrases -- the separate things this state is holding in mind. "
       "One per line, each starting with \"* \". They are combined afterwards, so each line "
       "should be a DIFFERENT part of the state rather than a rephrasing of the others.\n\n"
       "How it is judged. Your phrase is placed into a prompt of the form\n"
       '  Focus on the following idea: "<your phrase>" while writing the following phrase: '
       '"<a fixed unrelated sentence>"\n'
       "The model then writes that fixed sentence, and we read its internal state while it does "
       "so. You score well when that state matches the state you were given.\n\n"
       "So write what the model should be THINKING ABOUT -- not a description of a vector, and not "
       "a comment on the task. Natural, fluent English. At most 96 tokens. Output only the 4 lines."
       % C.INJ_CHAR)
txt = tok.apply_chat_template([{"role": "user", "content": JOB}], tokenize=False,
                              add_generation_prompt=True, enable_thinking=False)
PIDS = torch.tensor(tok.encode(txt, add_special_tokens=False), device=dev)
PLEN = PIDS.numel()
print("[t] prompt %d tok (training checkpoint used 229)" % PLEN, flush=True)

LANG = re.compile(r"\b(korean|chinese|japanese|norwegian|dutch|european|german|french|spanish|"
                  r"russian|swedish|danish|finnish|italian|portuguese|hindi|arabic|language|"
                  r"translat\w*|half.?translated|foreign)\b", re.I)


def gen(acts):
    B = acts.shape[0]
    HOOK["vec"] = acts.to(dev).float()
    try:
        g = model.generate(input_ids=PIDS.unsqueeze(0).expand(B, -1).contiguous(),
                           attention_mask=torch.ones(B, PLEN, device=dev, dtype=torch.long),
                           do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                           max_new_tokens=96, pad_token_id=EOS)
    finally:
        HOOK["vec"] = None
    out = []
    for row in g[:, PLEN:].tolist():
        cut = row.index(EOS) if EOS in row else len(row)
        out.append(tok.decode(row[:cut], skip_special_tokens=True).strip())
    return out


sets = {}
V = []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=4096, columns=["activation_vector"]):
    V.append(np.asarray(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in V) >= 4096:
        break
sets["TRAINING (fineweb prose L42)"] = torch.from_numpy(
    np.concatenate(V)[np.random.default_rng(0).choice(4096, N, replace=False)])
sets["PROBE (blogpost punctuation)"] = torch.from_numpy(
    np.load("/workspace/inv/data/holdout_blogpost.npy").astype("float32"))[:N]

res = {}
with torch.no_grad():
    for name, acts in sets.items():
        outs = []
        for s in range(0, acts.shape[0], 8):
            outs += gen(acts[s:s + 8])
        hits = [bool(LANG.search(o)) for o in outs]
        nb = [len([l for l in o.splitlines() if l.strip()]) for o in outs]
        res[name] = {"n": len(outs), "lang_rate": float(np.mean(hits)),
                     "mean_lines": float(np.mean(nb)),
                     "over4": float(np.mean([x > 4 for x in nb]))}
        print("\n%s  n=%d" % (name, len(outs)), flush=True)
        print("  language mentioned in %.0f%% of rollouts | mean lines %.2f | >4 lines %.0f%%"
              % (100 * res[name]["lang_rate"], res[name]["mean_lines"], 100 * res[name]["over4"]),
              flush=True)
        for o in outs[:2]:
            print("    %s" % o.replace("\n", " | ")[:190], flush=True)
json.dump(res, open("/workspace/inv/results/train_dist_lang.json", "w"), indent=1)
print("\nTRAIN_DIST_DONE", flush=True)
