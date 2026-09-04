"""Sanity check: do the two reliability estimators agree, on real activations?

TWO independent questions, both unverified until now:

(1) SAME-DATA AGREEMENT. Given one set of draws, does the variance-based statistic agree with the
    split-half cosine v3 ships? They are provably the same quantity up to the conversion
        rho = 1 - (n/(n-1))*(1 - ||mean of unit vectors||^2),  S = rho/(1-rho),
        reliability_8 = 8S/(8S+1)
    and I verified that identity numerically in simulation -- but never on real activations, where
    the noise is not the isotropic Gaussian the simulation assumed. If the two disagree here, the
    isotropy assumption is what breaks, and the k-extrapolation sqrt(kS/(kS+1)) goes with it.

(2) CROSS-GRID AGREEMENT. Does our measurement of an atom track v3's PUBLISHED reliability for the
    same atom? v3 used 16 templates against ONE fixed carrier, on raw L42. We use 6 templates x 6
    carriers, in J-space, with per-cell centering. Every survival/extrapolation number quoted so far
    assumes v3's distribution transfers to our measurement. If the correlation is weak, it does not,
    and the threshold table needs rebuilding from our own numbers.

Also reports the split-half estimator's OWN spread across different random splits -- the thing the
variance form is supposed to eliminate.
"""
import os

import modal

app = modal.App("celeste-agreement-check")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .apt_install("git")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "accelerate", "safetensors",
                    "sentencepiece", "pyarrow", "numpy", "huggingface_hub[hf_transfer]",
                    "einops", "flash-linear-attention", "scipy")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       .add_local_dir("/home/celeste/modlens_modal/src", "/root/src", copy=True))

WORKER = r'''
import glob, json, os, sys
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, "/root/src")
import inv_core as C
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

NPH = int(os.environ.get("NPHRASE", "1500"))
SPACE = os.environ.get("SPACE", "raw")   # v3 defines reliability on RAW L42, so raw is the fair test
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = model.model
inner.layers = nn.ModuleList(list(inner.layers[:43]))
J = C.load_jlens(42, dev)
HOOK = {"h": None}
inner.layers[42].register_forward_hook(
    lambda m, i, o: HOOK.__setitem__("h", o[0] if isinstance(o, tuple) else o))
G = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, 42, J, dev)
NCELL = G.n_tpl * G.n_car
print("[grid] %dt x %dc = %d cells | space=%s" % (G.n_tpl, G.n_car, NCELL, SPACE), flush=True)

labs, rels = [], []
for sh in sorted(glob.glob("/vol/thinkies_v3/thinkies_v3-*-of-00007.parquet")):
    for b in pq.ParquetFile(sh).iter_batches(batch_size=8192, columns=["label","reliability"]):
        labs += b.column("label").to_pylist(); rels += b.column("reliability").to_pylist()
        if len(labs) >= 80000: break
    break
rng = np.random.default_rng(0)
# stratify across the reliability range so the correlation is not driven by one narrow band
rels_a = np.array(rels, dtype="float32")
sel = []
for lo, hi in ((0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,1.01)):
    idx = np.nonzero((rels_a>=lo)&(rels_a<hi))[0]
    if len(idx): sel += list(rng.choice(idx, min(NPH//5, len(idx)), replace=False))
sel = np.array(sel)
PH  = [labs[i] for i in sel]
REL = rels_a[sel]
print("[data] %d atoms, published reliability %.3f-%.3f" % (len(PH), REL.min(), REL.max()), flush=True)

@torch.no_grad()
def read_cells(strings, batch=48):
    ids = {s: (tok(s, add_special_tokens=False).input_ids[:20] or
               tok(" the", add_special_tokens=False).input_ids) for s in set(strings)}
    pos = {s: k for k, s in enumerate(strings)}
    buckets = {}
    for s in strings: buckets.setdefault(len(ids[s]), []).append(s)
    out = np.zeros((len(strings), NCELL, J.shape[0]), dtype="float32")
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
                    _r = HOOK["h"].float()[:, -cell["ncar"]:, :].mean(1)
                    v = (_r @ J.T if SPACE == "jspace" else _r).cpu().numpy()
                    for k, s in enumerate(ch): out[pos[s], ci] = v[k]
            ci += 1
        print("   carrier %d/%d" % (c+1, G.n_car), flush=True)
    return out

X = []
CH = 300
for a in range(0, len(PH), CH):
    X.append(read_cells(PH[a:a+CH]))
    print("[read] %d/%d" % (min(a+CH, len(PH)), len(PH)), flush=True)
X = np.concatenate(X, 0)

MU = X.mean(axis=0, keepdims=True)          # per-cell mean over atoms
Z = X - MU
Z /= np.linalg.norm(Z, axis=-1, keepdims=True).clip(1e-8)
n = Z.shape[1]

# --- estimator 1: variance of the normalized draws ---
m = Z.mean(axis=1)
V = 1.0 - (m*m).sum(-1)
rho_var = 1.0 - (n/(n-1.0))*V
S_var = np.clip(rho_var, 1e-6, 1-1e-6); S_var = S_var/(1-S_var)

# --- estimator 2: split-half cosine, many random splits ---
rs = np.random.default_rng(1)
halves = []
for _ in range(50):
    p = rs.permutation(n)
    a = Z[:, p[:n//2]].mean(1); b = Z[:, p[n//2:]].mean(1)
    halves.append((a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)).clip(1e-9))
H = np.stack(halves, 1)                      # [B, 50]
r_half_mean = H.mean(1); r_half_sd = H.std(1)
k = n//2
S_half = np.clip(r_half_mean, 1e-6, 1-1e-6); S_half = (S_half/(1-S_half))/k

def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
def spear(a, b):
    from scipy.stats import rankdata
    return pear(rankdata(a), rankdata(b))

print("\n(1) SAME-DATA AGREEMENT  variance-form vs split-half, on identical draws")
print("    S from variance   mean %.4f" % S_var.mean())
print("    S from split-half mean %.4f  (averaged over 50 random splits)" % S_half.mean())
print("    Pearson r(S_var, S_half)  = %.4f" % pear(S_var, S_half))
print("    median |relative diff|    = %.2f%%" % (100*np.median(np.abs(S_var-S_half)/np.maximum(S_half,1e-9))))
print("\n    split-half's OWN spread across splits: median sd %.4f on r (this is what the"
      % np.median(r_half_sd))
print("    variance form removes -- it has no split to choose)")

print("\n(2) CROSS-GRID AGREEMENT  our 6x6 %s measurement vs v3's PUBLISHED 16t x 1c raw value" % SPACE.upper())
print("    Pearson  r = %.4f" % pear(S_var, REL))
try:
    print("    Spearman r = %.4f" % spear(S_var, REL))
except Exception: pass
print("    (correlation with rho rather than S: %.4f)" % pear(rho_var, REL))
print("\n    by published-reliability band, our mean rho:")
for lo, hi in ((0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,1.01)):
    s = (REL>=lo)&(REL<hi)
    if s.sum() > 20:
        print("      %.2f-%.2f  n=%4d   our rho %.4f  (implied 8-draw rel %.4f)"
              % (lo, hi, int(s.sum()), rho_var[s].mean(),
                 float((8*S_var[s]/(8*S_var[s]+1)).mean())))

json.dump({"n": int(len(PH)), "r_same_data": pear(S_var, S_half),
           "median_rel_diff_pct": float(100*np.median(np.abs(S_var-S_half)/np.maximum(S_half,1e-9))),
           "splithalf_own_sd_median": float(np.median(r_half_sd)),
           "r_cross_grid_S": pear(S_var, REL), "r_cross_grid_rho": pear(rho_var, REL),
           "rho_mean": float(rho_var.mean())},
          open("/vol/results_agreement_%s.json" % SPACE,"w"), indent=1)
print("\nAGREEMENT_DONE", flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=10800,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def check(nphrase: int = 1500, space: str = "raw"):
    import subprocess
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")
    open("/root/w.py", "w").write(WORKER)
    rc = subprocess.run(["python", "/root/w.py"],
                        env=dict(os.environ, NPHRASE=str(nphrase), SPACE=space)).returncode
    VOL.commit()
    return rc
