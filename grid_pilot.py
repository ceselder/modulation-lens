"""How many templates x carriers does a span actually need?

The CI on the consistency statistic depends on ONE unmeasured quantity: the effective dimension of
the nuisance noise. Simulated at rho=0.25, the 95% CI half-width for 16 draws is 0.005 if the noise
is isotropic in 5120 dims but 0.046 if it lives in 16 dims -- a 9x swing that decides whether the
production harvest needs 4 cells or 32. Guessing it wrong either wastes most of the compute or
produces a filter that cannot resolve the atoms it is selecting.

It also fixes a real gap in v3. That harvest varied 16 TEMPLATES against a FIXED carrier, so the
carrier contribution never averaged out and is invisible to the published `reliability`. This
project separately measured carrier effects as LARGER than template effects (sd 0.036 / range 0.119
across 8 carriers). Our reward averages 6 carriers, so the dictionary vectors and the reward's
targets do not even have the same nuisance structure.

So: read every one of the 36 (template, carrier) cells for a sample of phrases, keep the per-cell
vectors, and decompose. Output is a two-way variance split plus the effective dimension of each
component, which together determine the minimum grid for any target CI:

    Var(grid mean) ~ V_T/n_t + V_C/n_c + V_E/(n_t*n_c)
"""
import os

import modal

app = modal.App("celeste-grid-pilot")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .apt_install("git")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "peft==0.19.1", "accelerate",
                    "safetensors", "sentencepiece", "pyarrow", "numpy",
                    "huggingface_hub[hf_transfer]", "einops", "flash-linear-attention")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       .add_local_dir("/home/celeste/modlens_modal/src", "/root/src", copy=True))

WORKER = r'''
import glob, json, os, sys
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, "/root/src")
import inv_core as C
from transformers import AutoModelForCausalLM, AutoTokenizer

NPHRASE = int(os.environ.get("NPHRASE", "1200"))
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = model.model
J = C.load_jlens(42, dev)
HOOK = {"h": None}
inner.layers[42].register_forward_hook(
    lambda m, i, o: HOOK.__setitem__("h", o[0] if isinstance(o, tuple) else o))

G = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, 42, J, dev)
NT, NC = G.n_tpl, G.n_car
print("[grid] %d templates x %d carriers = %d cells | sig %s" % (NT, NC, NT*NC, G.sig()[:10]), flush=True)

# phrases: real atom labels, so results are comparable to the published reliability column
labs, rels = [], []
for sh in sorted(glob.glob("/vol/thinkies_v3/thinkies_v3-*-of-00007.parquet")):
    for b in pq.ParquetFile(sh).iter_batches(batch_size=8192, columns=["label","reliability"]):
        labs += b.column("label").to_pylist()
        rels += b.column("reliability").to_pylist()
        if len(labs) >= 60000: break
    break
rng = np.random.default_rng(0)
sel = rng.choice(len(labs), NPHRASE, replace=False)
PH  = [labs[i] for i in sel]
REL = np.array([rels[i] for i in sel], dtype="float32")
print("[data] %d phrases, published reliability mean %.3f" % (len(PH), REL.mean()), flush=True)

# ---- per-cell read (Grid.read averages templates within a carrier; we need each cell) ----
@torch.no_grad()
def read_cells(strings, max_tok=20, batch=48):
    ids_of = {}
    for s in strings:
        t = tok(s, add_special_tokens=False).input_ids[:max_tok]
        ids_of[s] = t or tok(" the", add_special_tokens=False).input_ids
    buckets = {}
    for s, t in ids_of.items():
        buckets.setdefault(len(t), []).append(s)
    out = np.zeros((len(strings), NC, NT, J.shape[0]), dtype="float32")
    pos = {s: k for k, s in enumerate(strings)}
    for ci in range(NC):
        for ti, S in enumerate(G.cells[ci]):
            pre  = torch.tensor(S["pre"],  device=dev)
            post = torch.tensor(S["post"], device=dev)
            for _, grp in buckets.items():
                for a in range(0, len(grp), batch):
                    ch = grp[a:a+batch]
                    mid = torch.tensor([ids_of[s] for s in ch], device=dev)
                    B = mid.shape[0]
                    model(input_ids=torch.cat([pre.unsqueeze(0).expand(B,-1), mid,
                                               post.unsqueeze(0).expand(B,-1)], dim=1))
                    v = HOOK["h"].float()[:, -S["ncar"]:, :].mean(1) @ J.T
                    vv = v.cpu().numpy()
                    for k, s in enumerate(ch):
                        out[pos[s], ci, ti] = vv[k]
        print("   carrier %d/%d done" % (ci+1, NC), flush=True)
    return out

CH = 150
parts = []
for a in range(0, len(PH), CH):
    parts.append(read_cells(PH[a:a+CH]))
    print("[read] %d/%d phrases" % (min(a+CH, len(PH)), len(PH)), flush=True)
X = np.concatenate(parts, 0)                      # [P, NC, NT, D]
P, _, _, D = X.shape
print("[read] tensor %s" % (X.shape,), flush=True)

# ---- per-CELL centering: each (carrier,template) cell has its own mean over phrases ----
# This is the analogue of v3's per-template centering, extended to both axes. Without it raw
# activations sit at cosine ~0.996 and every statistic saturates.
X = X - X.mean(axis=0, keepdims=True)
# directions: unit-normalize every cell so one cell's larger activations cannot dominate
X /= np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-8)

m   = X.mean(axis=(1, 2), keepdims=True)                 # per-phrase grand mean
a_t = X.mean(axis=1, keepdims=True) - m                   # template effect  [P,1,NT,D]
b_c = X.mean(axis=2, keepdims=True) - m                   # carrier  effect  [P,NC,1,D]
e   = X - m - a_t - b_c                                   # residual

V_T = float((a_t ** 2).sum(-1).mean())
V_C = float((b_c ** 2).sum(-1).mean())
V_E = float((e   ** 2).sum(-1).mean())
V_total = float(((X - m) ** 2).sum(-1).mean())
print("\n[variance decomposition]  (squared length, direction units)")
for n, v in (("template", V_T), ("carrier", V_C), ("residual", V_E)):
    print("   %-9s %.5f  (%.1f%% of total %.5f)" % (n, v, 100*v/V_total, V_total))

def d_eff(M):
    """participation ratio of the covariance spectrum = effective independent dimensions."""
    Z = M.reshape(-1, M.shape[-1])
    Z = Z - Z.mean(0, keepdims=True)
    if Z.shape[0] > 6000: Z = Z[np.random.default_rng(0).choice(Z.shape[0], 6000, replace=False)]
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())

dt, dc, de = d_eff(a_t.squeeze(1)), d_eff(b_c.squeeze(2)), d_eff(e)
print("\n[effective dimension]  (D_eff; isotropic in 5120 dims would be ~5120)")
print("   template effects %8.1f   (at most NT-1 = %d)" % (dt, NT-1))
print("   carrier  effects %8.1f   (at most NC-1 = %d)" % (dc, NC-1))
print("   residual         %8.1f" % de)

print("\n[design] Var(grid mean) ~ V_T/n_t + V_C/n_c + V_E/(n_t*n_c)")
print("%-14s %-12s %-12s" % ("grid", "pred Var", "vs 6x6"))
base = V_T/NT + V_C/NC + V_E/(NT*NC)
for nt, nc in ((1,1),(2,2),(3,3),(4,4),(6,6),(2,6),(6,2),(4,8),(8,4)):
    v = V_T/nt + V_C/nc + V_E/(nt*nc)
    print("%-14s %-12.5f %-12.2fx  (%d cells)" % ("%dt x %dc" % (nt,nc), v, v/base, nt*nc))

json.dump({"V_T":V_T,"V_C":V_C,"V_E":V_E,"V_total":V_total,
           "d_eff_template":dt,"d_eff_carrier":dc,"d_eff_residual":de,
           "n_tpl":NT,"n_car":NC,"n_phrase":P,
           "published_reliability_mean":float(REL.mean())},
          open("/vol/results_grid_pilot.json","w"), indent=1)
print("\nGRID_PILOT_DONE", flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=10800,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def pilot(nphrase: int = 1200):
    import subprocess
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")
    open("/root/w.py", "w").write(WORKER)
    p = subprocess.run(["python", "/root/w.py"], env=dict(os.environ, NPHRASE=str(nphrase)))
    VOL.commit()
    return p.returncode
