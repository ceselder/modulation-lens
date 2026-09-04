#!/usr/bin/env python3
"""
Does the two-term objective rank bullshit below good readouts?

Scores four known phrase sets on the same blogpost positions:
  rl50    genuine contrastive readouts        -- should win
  rl350   'not pizza related wording but...'  -- should LOSE (it wins under r_state alone)
  copy    the verbatim source span            -- should be penalised, it is not a readout
  generic one fixed bland phrase for all      -- should be near zero

r_state = cos(J@(read-PMU), J@(h-AMU))                      the old objective
r_text  = logP(own span | reader(p)) - mean logP(other spans | reader(p))
total   = r_state + lam*r_text - copy*jaccard(p, own span)
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

LAM, COPY, N = 1.0, 0.5, 24
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
L42 = {}
model.model.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
J = C.load_jlens(42, dev)
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED[:3], 42, J, dev)
PMU = torch.tensor(np.load("/workspace/inv/ckpts/rl/pmu.npy"), device=dev)
V = []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=4096, columns=["activation_vector"]):
    V.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in V) >= 20000:
        break
AMU = torch.from_numpy(np.concatenate(V)[:20000]).mean(0).to(dev) @ J.T

d = json.load(open("/workspace/inv/results/blogpost_readouts.json"))
key = lambda r: (r["para"], r["i"])
g50 = {key(r): r for r in d["runs"]["rl50"]["rows"]}
g350 = {key(r): r for r in d["runs"]["rl350"]["rows"]}
keys = [k for k in g50 if k in g350][:N]
paras = [x.strip() for x in open("/workspace/inv/data/blogpost.txt").read().split("\n\n")
         if x.strip()]
CPRE, CPOST = C.chat_wrap_ids(tok)
DELIM = set()
for _s in ". , ! ? ; : ...".split():
    for _c in (_s, " " + _s):
        _i = tok(_c, add_special_tokens=False).input_ids
        if len(_i) == 1:
            DELIM.add(_i[0])
TGT, SPAN = {}, {}
with torch.no_grad():
    for k in keys:
        pi, ti = k
        ids = tok(paras[pi], add_special_tokens=False, truncation=True, max_length=256).input_ids
        ti = min(ti, len(ids) - 1)
        model(input_ids=torch.tensor([CPRE + ids + CPOST], device=dev))
        H = L42["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
        t = (H[ti] @ J.T) - AMU
        TGT[k] = t / t.norm().clamp(min=1e-8)
        lo = 0
        for j in range(ti - 1, max(-1, ti - 17), -1):
            if ids[j] in DELIM:
                lo = j + 1
                break
        SPAN[k] = tok.decode(ids[max(lo, ti - 16):ti + 1]).strip()
spans = [SPAN[k] for k in keys]
print("[n] %d positions | lam_text %.1f | copy_penalty %.1f\n" % (len(keys), LAM, COPY), flush=True)

READER = ('A language model was reading some text. Someone who could see its internal state '
          'describes what it had in mind as: "%s"\n\nThe text it was reading was:')


@torch.no_grad()
def suff(phrases):
    out = np.zeros((len(phrases), len(spans)), dtype="float32")
    pairs, meta = [], []
    for pi_, ph in enumerate(phrases):
        pre = tok(READER % ph, add_special_tokens=False).input_ids
        for sj, sp in enumerate(spans):
            st = tok(" " + sp, add_special_tokens=False).input_ids[:32]
            if st:
                pairs.append((pre, st)); meta.append((pi_, sj))
    for a0 in range(0, len(pairs), 48):
        blk = pairs[a0:a0 + 48]
        T = max(len(x) + len(y) for x, y in blk)
        ids = torch.full((len(blk), T), tok.eos_token_id, device=dev, dtype=torch.long)
        msk = torch.zeros((len(blk), T), device=dev, dtype=torch.bool)
        for kk, (x, y) in enumerate(blk):
            ids[kk, :len(x)] = torch.tensor(x, device=dev)
            ids[kk, len(x):len(x) + len(y)] = torch.tensor(y, device=dev)
            msk[kk, len(x):len(x) + len(y)] = True
        lg = torch.log_softmax(model(input_ids=ids).logits[:, :-1].float(), -1)
        lp = lg.gather(2, ids[:, 1:].unsqueeze(2)).squeeze(2)
        mm = msk[:, 1:]
        per = (lp * mm).sum(1) / mm.sum(1).clamp(min=1)
        for kk in range(len(blk)):
            a_, b_ = meta[a0 + kk]
            out[a_, b_] = float(per[kk])
    return out


@torch.no_grad()
def state(phrases):
    v = GRID.read(model, phrases, L42, carrier=0, max_tok=24)
    return {p: (v[p] - PMU) / (v[p] - PMU).norm().clamp(min=1e-8) for p in phrases}


def jac(a, b):
    ta = set(tok(a, add_special_tokens=False).input_ids)
    tb = set(tok(b, add_special_tokens=False).input_ids)
    return len(ta & tb) / max(1, len(ta | tb))


SETS = {
    "rl50   (good readouts)": [g50[k]["phrase"] for k in keys],
    "rl350  (pizza boilerplate)": [g350[k]["phrase"] for k in keys],
    "copy   (verbatim span)": [SPAN[k] for k in keys],
    "generic (one bland phrase)": ["a passage of conversational writing"] * len(keys),
}
print("%-28s %9s %9s %9s %9s" % ("", "r_state", "r_text", "copy", "TOTAL"))
res = {}
for name, ph in SETS.items():
    uniq = sorted(set(ph))
    SV, SM = state(uniq), suff(uniq)
    ui = {p: i for i, p in enumerate(uniq)}
    rs, rt, cp, tot = [], [], [], []
    for i, k in enumerate(keys):
        p = ph[i]
        a = float(SV[p] @ TGT[k])
        row = SM[ui[p]]
        oth = [row[j] for j in range(len(keys)) if j != i]
        t_ = float(row[i] - np.mean(oth))
        c_ = jac(p, spans[i])
        rs.append(a); rt.append(t_); cp.append(c_); tot.append(a + LAM * t_ - COPY * c_)
    res[name] = {"r_state": float(np.mean(rs)), "r_text": float(np.mean(rt)),
                 "copy": float(np.mean(cp)), "total": float(np.mean(tot))}
    print("%-28s %+9.4f %+9.4f %9.3f %+9.4f"
          % (name, np.mean(rs), np.mean(rt), np.mean(cp), np.mean(tot)), flush=True)
print()
o = sorted(res.items(), key=lambda kv: -kv[1]["total"])
print("ranking under the NEW objective:  " + "  >  ".join(k.split()[0] for k, _ in o))
o2 = sorted(res.items(), key=lambda kv: -kv[1]["r_state"])
print("ranking under the OLD objective:  " + "  >  ".join(k.split()[0] for k, _ in o2))
json.dump(res, open("/workspace/inv/results/new_objective.json", "w"), indent=1)
print("\nNEW_OBJ_DONE", flush=True)
