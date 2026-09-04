"""Which space separates concepts best: raw L42, J-space, or whitened J-space?

Cosine in activation space is dominated by shared variance -- the whitener's own metadata says
k90 = 490, i.e. 490 of 5120 dimensions carry 90% of the variance. So plain cosine largely measures
alignment inside those 490 directions, and concept-specific structure in the low-variance tail is
drowned out. Whitening (Sigma^-1/2 after mean subtraction) equalises them.

This matters right now for a specific reason. Our 6x6 template grid was screened from 107-130
candidates UNDER PLAIN COSINE, and the consequence just showed up in measurement: our consistency
statistic reads 0.656-0.684 across the entire range of thinkies-v3's published reliability -- a
7x compression of that axis. Templates selected to maximise plain-cosine agreement produce
plain-cosine agreement for everything, which is right for a reward and useless for a filter.
Whitened cosine is NOT the metric they were selected under, so it should have more dynamic range.

Ground truth for "concept" is FineFineWeb's `domain` field (67 domains: aerospace, law, sports...).
Spans from one domain should be more similar to each other than to spans from another. Reported as:
  * within-domain vs across-domain mean cosine, and the gap
  * d-prime = (mu_within - mu_across) / pooled sd   <- scale-free separation
  * AUC of a same-domain-vs-different-domain classifier using cosine alone
  * the dynamic range of the per-span consistency statistic (rho), which is what a filter needs
"""
import os

import modal

app = modal.App("celeste-space-separation")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .apt_install("git")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "accelerate", "safetensors",
                    "sentencepiece", "pyarrow", "numpy", "huggingface_hub[hf_transfer]",
                    "einops", "flash-linear-attention")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       .add_local_dir("/home/celeste/modlens_modal/src", "/root/src", copy=True))

WORKER = r'''
import glob, json, os, sys, collections
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, "/root/src")
import inv_core as C
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

NDOM = int(os.environ.get("NDOM", "20"))
PER  = int(os.environ.get("PER", "150"))
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = m.model
inner.layers = nn.ModuleList(list(inner.layers[:43]))
J = C.load_jlens(42, dev)
HOOK = {"h": None}
inner.layers[42].register_forward_hook(
    lambda mm, i, o: HOOK.__setitem__("h", o[0] if isinstance(o, tuple) else o))
G = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, 42, J, dev)
NCELL = G.n_tpl * G.n_car
print("[grid] %d cells" % NCELL, flush=True)

MUw = torch.tensor(np.load("/root/wh_mu.npy"), device=dev)
Ws  = {k: torch.tensor(np.load("/root/wh_%s.npy" % k), device=dev) for k in ("r001", "r01")}
print("[whitener] mu |%.2f|, W cond r001/r01 loaded" % float(MUw.norm()), flush=True)

# domain-labelled spans from the FineFineWeb mining
rows = collections.defaultdict(list)
for f in sorted(glob.glob("/vol/spans_ffw/ffw-*.parquet"))[:60]:
    t = pq.read_table(f, columns=["span", "domain"]).to_pydict()
    for sp, dm in zip(t["span"], t["domain"]):
        if len(rows[dm]) < PER * 3: rows[dm].append(sp)
doms = [d for d, v in sorted(rows.items(), key=lambda kv: -len(kv[1]))[:NDOM]]
rng = np.random.default_rng(0)
SPANS, LAB = [], []
for d in doms:
    pick = rng.choice(len(rows[d]), min(PER, len(rows[d])), replace=False)
    for i in pick: SPANS.append(rows[d][i]); LAB.append(d)
LAB = np.array(LAB)
print("[data] %d spans across %d domains: %s" % (len(SPANS), len(doms), ", ".join(doms)), flush=True)

@torch.no_grad()
def read_cells(strings, batch=48):
    ids = {s: (tok(s, add_special_tokens=False).input_ids[:24] or
               tok(" the", add_special_tokens=False).input_ids) for s in set(strings)}
    pos = {s: k for k, s in enumerate(strings)}
    buckets = {}
    for s in strings: buckets.setdefault(len(ids[s]), []).append(s)
    out = np.zeros((len(strings), NCELL, 5120), dtype="float32")
    ci = 0
    for c in range(G.n_car):
        for t in range(G.n_tpl):
            cell = G.cells[c][t]
            pre  = torch.tensor(cell["pre"],  device=dev)
            post = torch.tensor(cell["post"], device=dev)
            for _, grp in buckets.items():
                for a in range(0, len(grp), batch):
                    ch = grp[a:a+batch]
                    mid = torch.tensor([ids[s] for s in ch], device=dev)
                    B = mid.shape[0]
                    inner(input_ids=torch.cat([pre.unsqueeze(0).expand(B,-1), mid,
                                               post.unsqueeze(0).expand(B,-1)], dim=1))
                    v = HOOK["h"].float()[:, -cell["ncar"]:, :].mean(1).cpu().numpy()
                    for k, s in enumerate(ch): out[pos[s], ci] = v[k]
            ci += 1
        print("   carrier %d/%d" % (c+1, G.n_car), flush=True)
    return out

X = []
for a in range(0, len(SPANS), 300):
    X.append(read_cells(SPANS[a:a+300]))
    print("[read] %d/%d" % (min(a+300, len(SPANS)), len(SPANS)), flush=True)
X = torch.from_numpy(np.concatenate(X, 0)).to(dev)      # [N, NCELL, 5120] RAW
print("[read] %s" % (tuple(X.shape),), flush=True)

def project(Xr, space):
    """RAW cell reads -> the space we measure cosines in, INCLUDING the centering order.

    W was fitted to the covariance of (natural activations - MUw), so whitening expects data
    centered by MUw. But the pipeline also needs the per-CELL mean removed (raw activations are
    ~93% a shared constant). Applying both means the whitening no longer sees the centering it was
    fitted for, and the order is not obviously determined -- so all three orders are tested rather
    than one being assumed. Variant A is what was run first and returned a null.
    """
    if space == "raw":
        return Xr
    Xj = Xr @ J.T
    if space == "jspace":
        return Xj
    cellmu = Xj.mean(dim=0, keepdim=True)          # per-cell mean, over spans
    if space.startswith("whitenA"):                # -MUw -> W  (then -cellmu downstream)
        W = Ws["r001" if space.endswith("r001") else "r01"]
        return (Xj - MUw) @ W.T
    if space.startswith("whitenB"):                # -cellmu -> -MUw -> W
        W = Ws["r001" if space.endswith("r001") else "r01"]
        return ((Xj - cellmu) - MUw) @ W.T
    if space.startswith("whitenC"):                # -cellmu -> W   (skip MUw: already centered)
        W = Ws["r001" if space.endswith("r001") else "r01"]
        return (Xj - cellmu) @ W.T
    if space == "whitenD_raw":                     # raw-space whitener, fitted here
        return (Xr - RAWMU) @ RAWW.T
    raise ValueError(space)

def auc(pos, neg):
    a = torch.cat([pos, neg]); y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    o = torch.argsort(a); r = torch.empty_like(a); r[o] = torch.arange(len(a), dtype=a.dtype, device=a.device)
    n1 = pos.numel(); n0 = neg.numel()
    return float((r[y == 1].sum() - n1*(n1-1)/2) / (n1*n0))

# The shipped whitener is J-space only (load_whitener refuses jtransformed != 1), so a raw-space
# variant has to be fitted here. NOTE: the covariance is 5120x5120 from N*NCELL pooled vectors, so
# it is underdetermined unless N*NCELL >> 5120 -- ridge is doing real work and this variant is the
# least trustworthy of the set. Reported anyway because "is it J or is it whitening" is otherwise
# unanswerable: the shipped whitener confounds the two.
_P = X.reshape(-1, X.shape[-1])
RAWMU = _P.mean(0, keepdim=True)
_Pc = (_P - RAWMU)
_C = (_Pc.T @ _Pc) / (_Pc.shape[0] - 1)
_lam = 0.1 * torch.diagonal(_C).mean()
_ev, _V = torch.linalg.eigh(_C.double() + _lam.double() * torch.eye(_C.shape[0], device=dev, dtype=torch.float64))
RAWW = (_V @ torch.diag(_ev.clamp(min=1e-8).rsqrt()) @ _V.T).float()
print("[raw whitener] fitted on %d pooled vectors, ridge %.4f (UNDERDETERMINED: %d samples for %d dims)"
      % (_P.shape[0], float(_lam), _P.shape[0], _C.shape[0]), flush=True)
del _P, _Pc, _C, _ev, _V

res = {}
for space in ("raw", "jspace", "whitenA_r001", "whitenA_r01",
              "whitenB_r001", "whitenB_r01", "whitenC_r001", "whitenC_r01", "whitenD_raw"):
    Z = project(X, space)
    # per-span consistency statistic (rho) -- what a filter would use
    Zc = Z - Z.mean(dim=0, keepdim=True)              # per-cell mean over spans
    Zn = Zc / Zc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mvec = Zn.mean(dim=1)
    V = 1.0 - (mvec*mvec).sum(-1)
    rho = 1.0 - (NCELL/(NCELL-1.0))*V
    # span-level embedding = grid mean, for the concept-separation test
    E = mvec / mvec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    Cm = E @ E.T
    N = Cm.shape[0]
    same = torch.tensor(LAB[:, None] == LAB[None, :], device=dev)
    off  = ~torch.eye(N, dtype=torch.bool, device=dev)
    w = Cm[same & off]; a_ = Cm[(~same) & off]
    dp = float((w.mean() - a_.mean()) / torch.sqrt(0.5*(w.var() + a_.var())))
    res[space] = {"rho_mean": float(rho.mean()), "rho_sd": float(rho.std()),
                  "rho_p5": float(torch.quantile(rho, 0.05)), "rho_p95": float(torch.quantile(rho, 0.95)),
                  "within": float(w.mean()), "across": float(a_.mean()),
                  "gap": float(w.mean()-a_.mean()), "dprime": dp, "auc": auc(w, a_)}
    print("\n[%s]" % space, flush=True)
    print("   concept separation: within %.4f  across %.4f  gap %+.4f  d' %.3f  AUC %.4f"
          % (res[space]["within"], res[space]["across"], res[space]["gap"], dp, res[space]["auc"]), flush=True)
    print("   rho (filter axis) : mean %.4f  sd %.4f  p5-p95 %.4f-%.4f  (range %.4f)"
          % (res[space]["rho_mean"], res[space]["rho_sd"], res[space]["rho_p5"],
             res[space]["rho_p95"], res[space]["rho_p95"]-res[space]["rho_p5"]), flush=True)

print("\n%-14s %8s %8s %8s | %9s %9s" % ("space","gap","d'","AUC","rho sd","rho range"))
for k, v in sorted(res.items(), key=lambda kv: -kv[1]["auc"]):
    print("%-14s %8.4f %8.3f %8.4f | %9.4f %9.4f"
          % (k, v["gap"], v["dprime"], v["auc"], v["rho_sd"], v["rho_p95"]-v["rho_p5"]))
json.dump({"n_spans": len(SPANS), "domains": doms, "spaces": res},
          open("/vol/results_space_separation.json","w"), indent=1)
print("\nSPACESEP_DONE", flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=10800)
def run(ndom: int = 20, per: int = 150):
    import subprocess
    import numpy as np
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")
    z = np.load("/vol/data/natural_whitener_jspace.npz")
    np.save("/root/wh_mu.npy", z["mu"])
    np.save("/root/wh_r001.npy", z["W_ridge0.01"])
    np.save("/root/wh_r01.npy", z["W_ridge0.1"])
    open("/root/w.py", "w").write(WORKER)
    rc = subprocess.run(["python", "/root/w.py"],
                        env=dict(os.environ, NDOM=str(ndom), PER=str(per))).returncode
    VOL.commit()
    return rc
