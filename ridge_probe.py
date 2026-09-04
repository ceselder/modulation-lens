"""Ridge baseline: label text -> atom vector, and does `reliability` predict predictability?

Two things this settles cheaply, before any GPU time goes into a 27B AR:
  1. Reproduces the prior 0.685-cosine ridge result on these exact vectors, so the AR has a
     concrete bar to clear rather than a remembered one.
  2. Held-out cosine BINNED BY RELIABILITY BAND. The plan is to filter the AR's training set by
     reliability, which only makes sense if high-reliability atoms are genuinely more predictable
     from their text. The dataset card warns reliability measures template agreement, NOT how
     uniquely the label pins the vector down -- so this is the check that the filter does what we
     want it to do.

Features are MEAN INPUT EMBEDDINGS of the label's tokens (5120-d), chosen because the earlier
finding was that bag-of-tokens ties embeddings closely: v(phrase) ~ sum of w(token). So this is
both the cheap baseline and a direct test of the additivity claim. Ridge is then 5120 -> 5120,
solved in closed form on GPU.

Only the embed_tokens tensor is read from the checkpoint -- loading the full 27B would cost 54GB
for a matmul we do not need.
"""
import os

import modal

app = modal.App("celeste-ridge-probe")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "safetensors", "pyarrow", "numpy",
                    "huggingface_hub")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}))

WORKER = r'''
import glob, json, os, sys
import numpy as np, pyarrow.parquet as pq, torch
from safetensors import safe_open
from transformers import AutoTokenizer

NROW = int(os.environ.get("NROW", "300000"))
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

# ---- embeddings only ----
E = None
for f in sorted(glob.glob("/vol/.hf_home/hub/models--Qwen--Qwen3.6-27B/snapshots/*/*.safetensors")):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if k.endswith("embed_tokens.weight"):
                E = sf.get_tensor(k).to(dev, torch.float32); break
    if E is not None: break
assert E is not None, "embed_tokens not found"
print("[emb] %s" % (tuple(E.shape),), flush=True)

# ---- atoms ----
labs, vecs, rels = [], [], []
for sh in sorted(glob.glob("/vol/thinkies_v3/thinkies_v3-*-of-00007.parquet")):
    for b in pq.ParquetFile(sh).iter_batches(batch_size=16384,
                                            columns=["label","vector","reliability"]):
        labs += b.column("label").to_pylist()
        rels.append(np.asarray(b.column("reliability").to_numpy(zero_copy_only=False), dtype="float32"))
        vecs.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                               dtype="float32").reshape(-1, 5120))
        if sum(v.shape[0] for v in vecs) >= NROW: break
    if sum(v.shape[0] for v in vecs) >= NROW: break
Y = torch.from_numpy(np.concatenate(vecs)[:NROW]).to(dev)
R = np.concatenate(rels)[:NROW]
labs = labs[:NROW]
print("[data] %d atoms, rel mean %.3f" % (Y.shape[0], R.mean()), flush=True)

# ---- features: mean input embedding of the label's tokens ----
X = torch.zeros(len(labs), E.shape[1], device=dev)
B = 4096
for s in range(0, len(labs), B):
    chunk = labs[s:s+B]
    enc = tok(chunk, add_special_tokens=False)["input_ids"]
    for j, ids in enumerate(enc):
        if ids: X[s+j] = E[torch.tensor(ids, device=dev)].mean(0)
print("[feat] built", flush=True)

# targets are DIRECTIONS -- scale is discarded downstream, so fit on unit vectors
Yn = Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-8)
Xc = X - X.mean(0, keepdim=True)

n = Xc.shape[0]; ntr = int(0.9 * n)
g = torch.Generator(device="cpu").manual_seed(0)
perm = torch.randperm(n, generator=g).to(dev)
tr, te = perm[:ntr], perm[ntr:]

XtX = Xc[tr].T @ Xc[tr]
XtY = Xc[tr].T @ Yn[tr]
I = torch.eye(XtX.shape[0], device=dev)
best = None
for lam_mult in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
    lam = lam_mult * torch.diagonal(XtX).mean()
    W = torch.linalg.solve(XtX + lam * I, XtY)
    P = Xc[te] @ W
    P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-8)
    c = (P * Yn[te]).sum(1)
    m = float(c.mean())
    print("[ridge] lam_mult %.0e  held-out cos %.4f" % (lam_mult, m), flush=True)
    if best is None or m > best[0]: best = (m, lam_mult, W)
m, lam_mult, W = best
print("\n[BEST] held-out cos %.4f (lam_mult %.0e)" % (m, lam_mult), flush=True)

P = Xc[te] @ W
P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-8)
cos = (P * Yn[te]).sum(1).cpu().numpy()
rte = R[te.cpu().numpy()]
print("\nheld-out cosine BY RELIABILITY BAND  (implied target accuracy = sqrt(2r/(1+r)))")
print("%-14s %8s %10s %12s %10s" % ("band","n","ridge cos","implied max","ratio"))
out = {"overall": float(m), "lam_mult": lam_mult, "bands": []}
for lo, hi in ((0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,1.01)):
    sel = (rte>=lo)&(rte<hi)
    if sel.sum() < 30: continue
    r = float(rte[sel].mean()); cm = float(cos[sel].mean())
    imp = (2*r/(1+r))**0.5
    print("%.2f - %.2f   %8d %10.4f %12.4f %10.2f" % (lo,hi,int(sel.sum()),cm,imp,cm/imp))
    out["bands"].append({"lo":lo,"hi":hi,"n":int(sel.sum()),"rel_mean":r,
                         "ridge_cos":cm,"implied_max":imp,"ratio":cm/imp})
json.dump(out, open("/vol/results_ridge.json","w"), indent=1)
print("\nRIDGE_DONE", flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=5400)
def probe(nrow: int = 300000):
    import subprocess
    open("/root/w.py", "w").write(WORKER)
    p = subprocess.run(["python", "/root/w.py"], env=dict(os.environ, NROW=str(nrow)))
    VOL.commit()
    return p.returncode
