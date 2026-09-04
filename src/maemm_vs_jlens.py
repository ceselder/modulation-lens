#!/usr/bin/env python3
"""Does MAEMM tell you more about a CONFUSED modulation vector than its label or its J-lens?

The claim under test: when a modulation vector is "confused", MAEMM readouts carry more information
about what it encodes than either the stored description or J-lensing it.

Operationalisation. "Confused" is not eyeballed: thinkies-v3 ships a `reliability` column (split-half
agreement across the 16 harvest templates), so low reliability = the templates disagreed about the
direction = confused. Vectors are bucketed by it.

Three readouts per vector:
  label   -- the stored phrase the vector was harvested for
  jlens   -- top-k tokens of (v @ J[42].T) through the unembedding
  maemm   -- ceselder/qwen36-27b-maemm-inverter, its OWN prompt/marker/injection (norm-matched add
             of unit(v) at layer 1 on a trailing " ?" marker), sampled

Metric: read each readout text back through the modulation grid (6 templates x 6 carriers, all 36
cells, PMU-centred, J-space) and take cosine to the vector. This asks "how much of this direction
does the text actually re-induce", which is the same geometry the modulation lens is trained in.

KNOWN CONFOUND, stated because it cuts against the claim's rival: this metric scores TEXT by its
modulation effect, so a bag of J-lens tokens is at a structural disadvantage versus a fluent phrase
regardless of how informative it is to a human. A mismatched-target control (`mis`) is computed for
every arm so we can separate "encodes this vector" from "reads well through a grid at all".
"""
import argparse, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

p = argparse.ArgumentParser()
p.add_argument("--maemm", default="ceselder/qwen36-27b-maemm-inverter")
p.add_argument("--dict-dir", default="/workspace/thinkies/v3")
p.add_argument("--pmu", default="/workspace/inv/ckpts/rl_v2_plain/pmu_db4a6b8ee6.npy")
p.add_argument("--n-per-bucket", type=int, default=40)
p.add_argument("--samples", type=int, default=4, help="maemm rollouts per vector")
p.add_argument("--topk", type=int, default=12, help="jlens tokens per vector")
p.add_argument("--max-new", type=int, default=32)
p.add_argument("--out", default="/workspace/inv/results/maemm_vs_jlens.json")
A = p.parse_args()
dev = "cuda"
EOS = 248046

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
J = C.load_jlens(42, dev)
PMU = torch.from_numpy(np.load(A.pmu)).to(dev).float()

# ---- MAEMM's own prompt, verbatim from mxf/prompts.py ----
MARKER = " ?"
INSTR = ("You are an interpretability research tool. Researchers have injected a single linear probe "
         "direction from this model's own layer-42 residual stream. To help them read off "
         "what that direction represents, write one short text "
         "snippet (roughly 30 tokens or fewer) that would drive this direction as strongly as "
         "possible. Output only the snippet itself, with no explanation, preamble, or quotation "
         "marks. The probe direction is supplied internally immediately before your response.")
_p = tok.apply_chat_template([{"role": "user", "content": INSTR}], tokenize=True,
                             add_generation_prompt=True, enable_thinking=False)
_p = _p["input_ids"] if hasattr(_p, "keys") else _p
while isinstance(_p[0], list):
    _p = _p[0]
MPROMPT = list(_p) + tok.encode(MARKER, add_special_tokens=False)
MPOS = len(MPROMPT) - 1
assert len(tok.encode(MARKER, add_special_tokens=False)) == 1, "marker must be single-token"
print("[m] maemm prompt %d tok, marker at %d (last)" % (len(MPROMPT), MPOS), flush=True)

VREF = {"v": None}
inner = base.model


def _hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    v = VREF["v"]
    if v is None or h.shape[1] <= 1:            # decode steps: injected at prefill already
        return out
    b = h[:, MPOS]
    nv = torch.nn.functional.normalize(v, dim=-1)
    h[:, MPOS] = b + (nv * b.norm(dim=-1, keepdim=True)).to(h.dtype)   # norm-matched ADD, coeff 1
    return out


inner.layers[1].register_forward_hook(_hook)
model = PeftModel.from_pretrained(base, A.maemm, adapter_name="maemm").eval()
GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, 42, J, dev)
L42 = {}
inner.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
WU = base.lm_head.weight.detach()

# ---- thinkies vectors bucketed by reliability ("confused" = low) ----
labs, vecs, rels = [], [], []
for sh in sorted(glob.glob(os.path.join(A.dict_dir, "thinkies_v3-*.parquet"))):
    for b in pq.ParquetFile(sh).iter_batches(batch_size=16384,
                                             columns=["label", "vector", "reliability"]):
        l = b.column("label").to_pylist()
        labs += l
        vecs.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                               dtype="float32").reshape(len(l), -1))
        rels += list(np.asarray(b.column("reliability").to_numpy(zero_copy_only=False),
                                dtype="float32"))
    if len(labs) >= 120000:
        break
V = np.concatenate(vecs); R = np.asarray(rels)
q = np.nanquantile(R, [0.05, 0.25, 0.75, 0.95])
print("[m] %d vectors | reliability quantiles 5/25/75/95: %s" % (len(labs), np.round(q, 3)),
      flush=True)
rng = np.random.default_rng(0)
buckets = {}
for name, lo, hi in (("confused", -1e9, q[1]), ("clear", q[2], 1e9)):
    idx = np.where((R > lo) & (R <= hi))[0]
    buckets[name] = rng.choice(idx, min(A.n_per_bucket, len(idx)), replace=False)
    print("  %-9s n=%d  reliability %.3f-%.3f" % (name, len(buckets[name]),
          R[buckets[name]].min(), R[buckets[name]].max()), flush=True)


@torch.no_grad()
def maemm_read(vv):
    ids = torch.tensor([MPROMPT] * vv.shape[0], device=dev)
    VREF["v"] = vv
    try:
        model.set_adapter("maemm")
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                           temperature=0.9, top_p=0.95, max_new_tokens=A.max_new, pad_token_id=EOS)
    finally:
        VREF["v"] = None
    out = []
    for row in g[:, len(MPROMPT):].tolist():
        cut = row.index(EOS) if EOS in row else len(row)
        out.append(tok.decode(row[:cut], skip_special_tokens=True).strip())
    return out


@torch.no_grad()
def jlens_read(vv):
    t = vv @ J.T
    lg = t @ WU.T.float()
    out = []
    for i in range(t.shape[0]):
        ids = torch.topk(lg[i], A.topk).indices.tolist()
        out.append(" ".join(tok.decode([j]).strip() for j in ids))
    return out


res = {"buckets": {}, "config": vars(A)}
for name, idx in buckets.items():
    tv = torch.from_numpy(V[idx].astype("float32")).to(dev)
    tj = tv @ J.T
    tj = tj / tj.norm(dim=1, keepdim=True).clamp(min=1e-8)
    reads = {"label": [labs[i] for i in idx], "jlens": [], "maemm": []}
    for s in range(0, len(idx), 8):
        reads["jlens"] += jlens_read(tv[s:s + 8])
        cands = [maemm_read(tv[s:s + 8]) for _ in range(A.samples)]
        # keep the MAEMM sample that best re-induces the direction (best-of-k, as its own eval does)
        reads["maemm"] += list(zip(*cands))
    rows = {}
    for arm in ("label", "jlens", "maemm"):
        txts = reads[arm]
        flat = sorted({x for t_ in txts for x in ((t_,) if isinstance(t_, str) else t_)})
        with model.disable_adapter():
            rv = GRID.read_all(model, flat, L42, max_tok=48, batch=128)
        own, mis = [], []
        for k in range(len(idx)):
            cand = txts[k] if isinstance(txts[k], str) else txts[k]
            cl = [cand] if isinstance(cand, str) else list(cand)
            best = max(float(((rv[c] - PMU) @ tj[k]) / (rv[c] - PMU).norm()) for c in cl)
            own.append(best)
            c0 = cl[0]
            mis += [float(((rv[c0] - PMU) @ tj[m]) / (rv[c0] - PMU).norm())
                    for m in range(len(idx)) if m != k][:8]
        rows[arm] = {"own": float(np.mean(own)), "mis": float(np.mean(mis)),
                     "disc": float(np.mean(own)) - float(np.mean(mis)),
                     "example": (txts[0] if isinstance(txts[0], str) else txts[0][0])[:100]}
        print("  [%s] %-6s own %.4f  mis %.4f  disc %.4f | %r"
              % (name, arm, rows[arm]["own"], rows[arm]["mis"], rows[arm]["disc"],
                 rows[arm]["example"][:70]), flush=True)
    res["buckets"][name] = rows
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(res, open(A.out, "w"), indent=1)
print("\nCLAIM TEST: maemm-minus-jlens disc should be LARGER in the confused bucket.")
for n in res["buckets"]:
    b = res["buckets"][n]
    print("  %-9s maemm-jlens %+.4f | maemm-label %+.4f"
          % (n, b["maemm"]["disc"] - b["jlens"]["disc"], b["maemm"]["disc"] - b["label"]["disc"]))
print("MAEMM_VS_JLENS_DONE", flush=True)
