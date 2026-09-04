#!/usr/bin/env python3
"""Score saved checkpoints on a properly-sized held-out composition probe.

The in-run probe is 32 activations: per-item composition cosine spreads by ~0.15, so its standard
error is ~0.027 and it cannot resolve anything under ~0.05. Every movement after step 25 of the
big-batch run sat inside that. This measures the SAME quantity the reward optimises -- four bullets
generated for a held-out activation, read back through the full 36-cell grid, combined by exact
non-negative least squares -- on rows that training never saw, and reports a standard error so the
differences can actually be called.

Default --n 512 gives SEM ~0.007 (4x tighter than the in-run probe) at ~7 min per checkpoint;
--n 2048 gives ~0.003 at ~27 min. Both dominated by the grid read, which is 36 cells per bullet.
"""
import argparse, glob, json, os, re, sys, time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True, help="adapter dirs (iter_* or final)")
ap.add_argument("--probe-npy", default="/workspace/inv/data/holdout_fresh.npy")
ap.add_argument("--pmu", default="/workspace/inv/ckpts/rl_nnols4_b64/pmu_db4a6b8ee6.npy")
ap.add_argument("--data", default="/root/data/prose_L42_500k.parquet",
                help="pool whose mean is the trainer's AMU; must match the training run")
ap.add_argument("--n-pool", type=int, default=500000)
ap.add_argument("--min-words", type=int, default=4)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--bullets", type=int, default=4)
ap.add_argument("--bullet-max-tok", type=int, default=10)
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--n-carriers", type=int, default=6)
ap.add_argument("--read-batch", type=int, default=256)
ap.add_argument("--gen-batch", type=int, default=64)
ap.add_argument("--inject", default="karvonen", choices=["replace", "karvonen"])
ap.add_argument("--out", default="/workspace/inv/results/ckpt_probe.jsonl")
A = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = base.model
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
    new = C.inject_at_marker(ids, resid, vec, INJ, LEFT, RIGHT, mode=A.inject)
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


inner.layers[1].register_forward_hook(_inj)
inner.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))

J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[: A.n_carriers], 42, J, dev)
PMU = torch.from_numpy(np.load(A.pmu)).to(dev).float()

# AMU: the trainer's target-side centre = mean of the pool it trained on, after min_words
import pyarrow.parquet as pq
acc, labs = [], []
for b in pq.ParquetFile(A.data).iter_batches(batch_size=8192,
                                             columns=["activation_vector", "label"]):
    d = b.to_pydict()
    acc.append(np.asarray(d["activation_vector"], dtype="float32"))
    labs += d["label"]
    if sum(len(x) for x in acc) >= A.n_pool:
        break
V = np.concatenate(acc)
keep = [i for i, l in enumerate(labs) if len(l.split()) >= A.min_words][: A.n_pool]
AMU = torch.from_numpy(V[keep].mean(0)).to(dev) @ J.T
del acc, V
print("[i] |AMU| %.2f |PMU| %.2f | grid %s" % (float(AMU.norm()), float(PMU.norm()),
                                               GRID.sig()[:10]), flush=True)

ACT = torch.from_numpy(np.load(A.probe_npy).astype("float32"))[: A.n]
print("[i] %d held-out activations" % ACT.shape[0], flush=True)

_BUL = re.compile(r"^\s*(?:[*•\-–]|\d+[.)])\s+")


def split_bullets(text, n, max_tok):
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    marked = [l for l in lines if _BUL.match(l)]
    out = [_BUL.sub("", l).strip() for l in (marked if marked else lines)[:n]]
    out = [l for l in out if l] or [str(text).strip() or "the"]
    res = []
    for b in out:
        ids = tok(b, add_special_tokens=False).input_ids
        res.append((tok.decode(ids[:max_tok]).strip() if len(ids) > max_tok else b) or "the")
    return res


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


os.makedirs(os.path.dirname(A.out), exist_ok=True)
model = None
for ck in A.ckpts:
    pf = os.path.join(ck, "prompt.txt")
    if not os.path.exists(pf):
        pf = os.path.join(os.path.dirname(ck.rstrip("/")), "prompt.txt")
    JOB = open(pf).read()
    PIDS = torch.tensor(tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": JOB}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False), add_special_tokens=False), device=dev)
    PLEN = PIDS.numel()
    assert int((PIDS == INJ).sum()) == 1, "prompt must carry exactly one marker"
    if model is None:
        model = PeftModel.from_pretrained(base, ck, adapter_name="m").eval()
    else:
        model.load_adapter(ck, adapter_name="m", is_trainable=False)
    model.set_adapter("m")
    t0 = time.time()
    cos, nz = [], []
    with torch.no_grad():
        for s in range(0, ACT.shape[0], A.gen_batch):
            sub = ACT[s:s + A.gen_batch].to(dev).float()
            B = sub.shape[0]
            HOOK["vec"] = sub
            try:
                g = model.generate(input_ids=PIDS.unsqueeze(0).expand(B, -1).contiguous(),
                                   attention_mask=torch.ones(B, PLEN, device=dev, dtype=torch.long),
                                   do_sample=False, max_new_tokens=A.max_new,
                                   pad_token_id=248046)
            finally:
                HOOK["vec"] = None
            txts = tok.batch_decode(g[:, PLEN:], skip_special_tokens=True)
            bl = [split_bullets(t, A.bullets, A.bullet_max_tok) for t in txts]
            uniq = sorted({b for row in bl for b in row})
            with model.disable_adapter():
                cv = GRID.read_all(model, uniq, L42, max_tok=A.bullet_max_tok,
                                   batch=A.read_batch)
            for k in range(B):
                M = torch.stack([cv[b] - PMU for b in bl[k]])
                t = (sub[k] @ J.T) - AMU
                t = t / t.norm().clamp(min=1e-8)
                w = nnls_small(M, t)
                rec = w @ M
                cos.append(float((rec @ t) / rec.norm().clamp(min=1e-8)))
                nz.append(int((w > 1e-6).sum()))
    m = float(np.mean(cos))
    sem = float(np.std(cos, ddof=1) / np.sqrt(len(cos)))
    row = {"ckpt": ck, "n": len(cos), "probe_cos": round(m, 4), "sem": round(sem, 4),
           "sd": round(float(np.std(cos, ddof=1)), 4),
           "mean_bullets_used_by_nnls": round(float(np.mean(nz)), 3),
           "seconds": round(time.time() - t0, 1)}
    with open(A.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("[probe] %s -> %.4f +-%.4f (sd %.3f, nnls uses %.2f/%d, %.0fs)"
          % (os.path.basename(ck.rstrip("/")), m, sem, row["sd"],
             row["mean_bullets_used_by_nnls"], A.bullets, row["seconds"]), flush=True)
print("CKPT_PROBE_DONE", flush=True)
