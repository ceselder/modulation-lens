#!/usr/bin/env python3
"""
Readouts for blogpost punctuation positions, from several checkpoints, for the report.

Three checkpoints matter:
  sft/final          clean committed paraphrase, before any RL
  rl/iter_000050     the contrastive-negation device while it was still semantic
  rl/iter_000350     after the device hollowed into a fixed activation-independent prefix
"""
import argparse, collections, json, os, sys
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

ap = argparse.ArgumentParser()
ap.add_argument("--text-file", default="/workspace/inv/data/blogpost.txt")
ap.add_argument("--ckpts", default="/workspace/inv/ckpts/sft/final,"
                                  "/workspace/inv/ckpts/rl/iter_000050,"
                                  "/workspace/inv/ckpts/rl/iter_000350")
ap.add_argument("--names", default="sft,rl50,rl350")
ap.add_argument("--n-pos", type=int, default=96)
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--carriers", type=int, default=3)
ap.add_argument("--layer", type=int, default=42)
ap.add_argument("--max-new", type=int, default=16)
ap.add_argument("--whitener", default="/workspace/inv/data/meansub/natural_whitener_jspace.npz")
ap.add_argument("--acts-pool", default="/workspace/inv/data/prose_L42.parquet")
ap.add_argument("--out", default="/workspace/inv/results/blogpost_readouts.json")
A = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
INJ, LEFT, RIGHT = C.marker_ids(tok)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
J = C.load_jlens(A.layer, dev)
MU, Wm = C.load_whitener(A.whitener, "0.1", dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[: A.carriers], A.layer, J, dev)

# --- blogpost punctuation positions, chat-native ---
CPRE, CPOST = C.chat_wrap_ids(tok)
DELIM = {}
for s in ". , ! ? ; : ...".split():
    for c in (s, " " + s):
        i = tok(c, add_special_tokens=False).input_ids
        if len(i) == 1:
            DELIM[i[0]] = s
L42 = {}
base.model.layers[A.layer].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
paras = [x.strip() for x in open(A.text_file).read().split("\n\n") if x.strip()]
POS = []
with torch.no_grad():
    for pi, para in enumerate(paras):
        ids = tok(para, add_special_tokens=False, truncation=True, max_length=256).input_ids
        if not ids:
            continue
        base(input_ids=torch.tensor([CPRE + ids + CPOST], device=dev))
        H = L42["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
        strs = [tok.decode([i]) for i in ids]
        for k in range(len(ids)):
            if ids[k] in DELIM:
                POS.append({"para": pi, "i": k, "mark": DELIM[ids[k]],
                            "ctx": "".join(strs[max(0, k - 26):k + 1]),
                            "h": H[k].cpu().numpy()})
print("[r] %d punctuation positions in the blogpost" % len(POS), flush=True)
by = collections.defaultdict(list)
for r in POS:
    by[r["mark"]].append(r)
pick, want = [], A.n_pos
marks = sorted(by, key=lambda m: -len(by[m]))
while want > 0 and any(by[m] for m in marks):
    for m in marks:
        if by[m] and want > 0:
            pick.append(by[m].pop(len(by[m]) // 2))
            want -= 1
print("[r] sampled %d: %s" % (len(pick), dict(collections.Counter(p["mark"] for p in pick))),
      flush=True)

# target-side centre: the training pool's mean, same as the reward used
import pyarrow.parquet as pq
acc = []
for b in pq.ParquetFile(A.acts_pool).iter_batches(batch_size=4096, columns=["activation_vector"]):
    acc.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in acc) >= 20000:
        break
AMU = torch.from_numpy(np.concatenate(acc)[:20000]).mean(0).to(dev) @ J.T

JOBTXT = open("/workspace/inv/src/jobprompt.txt").read() if os.path.exists(
    "/workspace/inv/src/jobprompt.txt") else None
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

out = {"positions": [{k: v for k, v in p.items() if k != "h"} for p in pick], "runs": {}}
names = A.names.split(",")
for ci, cd in enumerate(A.ckpts.split(",")):
    name = names[ci]
    print("\n[r] === %s (%s) ===" % (name, cd), flush=True)
    m = PeftModel.from_pretrained(base, cd, adapter_name="d%d" % ci)
    m.set_adapter("d%d" % ci)
    inner = m.base_model.model.model
    HK = {"vec": None, "ids": None}
    h1 = inner.register_forward_pre_hook(
        lambda mod, a, kw: HK.__setitem__("ids", kw.get("input_ids", a[0] if a else None)),
        with_kwargs=True)
    def _inj(mod, a, o):
        r = o[0] if isinstance(o, tuple) else o
        if HK["vec"] is None or HK["ids"] is None: return o
        if tuple(HK["ids"].shape) != tuple(r.shape[:-1]) or not bool((HK["ids"] == INJ).any()):
            return o
        n = C.inject_at_marker(HK["ids"], r, HK["vec"], INJ, LEFT, RIGHT)
        return (n,) + tuple(o[1:]) if isinstance(o, tuple) else n
    h2 = inner.layers[1].register_forward_hook(_inj)

    gen_by = {}
    with torch.no_grad():
        for a in range(0, len(pick), 4):
            blk = pick[a:a + 4]
            v = torch.tensor(np.stack([p["h"] for p in blk]), device=dev).float()
            B = len(blk) * A.samples
            HK["vec"] = v.repeat_interleave(A.samples, 0)
            g = m.generate(input_ids=PIDS.unsqueeze(0).expand(B, -1).contiguous(),
                           attention_mask=torch.ones(B, PLEN, device=dev, dtype=torch.long),
                           max_new_tokens=A.max_new, do_sample=True, temperature=1.0,
                           top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
            HK["vec"] = None
            for j, t in enumerate(tok.batch_decode(g[:, PLEN:], skip_special_tokens=True)):
                gen_by.setdefault(a + j // A.samples, []).append(t.strip() or " the")
    h1.remove(); h2.remove()
    with m.disable_adapter():
        alls = sorted({t for v in gen_by.values() for t in v})
        vecs = GRID.read(m, alls, L42, carrier=0, max_tok=A.max_new)
    rows = []
    for k, p in enumerate(pick):
        t = ((torch.tensor(p["h"], device=dev) @ J.T) - AMU) @ Wm.T
        t = t / t.norm().clamp(min=1e-8)
        best, bs = None, -9
        for s in gen_by.get(k, []):
            a_ = (vecs[s] - torch.tensor(np.load("%s/../pmu.npy" % cd)
                  if os.path.exists("%s/../pmu.npy" % cd) else np.zeros(J.shape[0]),
                  device=dev, dtype=torch.float32)) @ Wm.T
            sc = float((a_ @ t) / a_.norm().clamp(min=1e-8))
            if sc > bs:
                best, bs = s, sc
        rows.append({"para": p["para"], "i": p["i"], "mark": p["mark"], "ctx": p["ctx"],
                     "phrase": best, "score": round(bs, 4)})
    out["runs"][name] = {"ckpt": cd, "rows": rows,
                         "mean": float(np.mean([r["score"] for r in rows]))}
    print("[r] %s mean %.4f" % (name, out["runs"][name]["mean"]), flush=True)
    for r in sorted(rows, key=lambda z: -z["score"])[:6]:
        print("    %.4f  %-5r ...%-40s -> %r" % (r["score"], r["mark"], r["ctx"][-40:],
                                                 r["phrase"][:46]), flush=True)
    m.delete_adapter("d%d" % ci)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(out, open(A.out, "w"), indent=1)
print("\nwrote %s\nBLOGPOST_READOUTS_DONE" % A.out, flush=True)
