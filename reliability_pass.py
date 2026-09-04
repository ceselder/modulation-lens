"""Stage 2: measure each span's consistency statistic, then THROW THE VECTORS AWAY.

For every span, read it in all (template x carrier) cells, and reduce those cells to a handful of
scalars. The vectors themselves are never stored -- that is what turns a 1 TB dataset into a ~1 GB
one, and they can always be recomputed from the span text.

Per span, with the per-cell-centered draws normalized to unit length:

    V   = 1 - ||mean of the unit vectors||^2      # variance within the group
    rho = 1 - (n/(n-1)) * V                      # == mean pairwise cosine over all n(n-1) pairs
    S   = rho / (1 - rho)                        # signal / noise
    acc_k = sqrt(k*S/(k*S+1))                    # accuracy of a k-cell mean, for ANY k

The CI is a leave-one-out JACKKNIFE over cells rather than an analytic formula. Analytic would need
the effective noise dimension, which the design pilot measured at ~39-55 (not isotropic-5120) and
which almost certainly varies per span. The jackknife costs no extra forwards and makes no isotropy
assumption -- it just recomputes rho 16 times, dropping one cell each time.

Grid is 4 templates x 4 carriers. The design pilot decomposed the variance as template 30.2% /
carrier 43.8% / residual 26.0%, and since each nuisance factor only averages over its OWN axis, a
near-square grid dominates: 4x4 is 2.35x better than thinkies-v3's 16-templates-x-1-carrier at
identical cost, and even 2x2 with FOUR cells beats it.

Model is truncated to 43 layers: we read layer 42, so layers 43-63 are dead compute (measured 1.24x,
less than the 1.49x the layer count implies, because ~40% of the time is not in the layers).
Batch 48 -- measured; 256 and 512 are ~2.4x SLOWER, likely the GDN kernel's (chunks, batch*heads)
grid, so parallelism comes from more GPUs, not bigger batches.

Resumable per output chunk, because a multi-day run WILL lose containers.
"""
import os

import modal

app = modal.App("celeste-reliability-pass")
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
import glob, json, os, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
sys.path.insert(0, "/root/src")
import inv_core as C
from transformers import AutoModelForCausalLM, AutoTokenizer

NT   = int(os.environ.get("NT", "4"))
NC   = int(os.environ.get("NC", "4"))
BATCH= int(os.environ.get("BATCH", "48"))
SPACE= os.environ.get("SPACE", "raw")   # "raw" (matches thinkies-v3) or "jspace" (matches the reward)
SHARD= int(os.environ["SHARD"]); NSHARD = int(os.environ["NSHARD"])
IN   = os.environ.get("IN_DIR", "/vol/spans_10m")
OUT  = os.environ.get("OUT_DIR", "/vol/reliability_10m")
CHUNK= int(os.environ.get("CHUNK", "20000"))
LIMIT= int(os.environ.get("LIMIT", "0"))     # cap the span list before sharding; 0 = all
dev = "cuda"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = model.model
import torch.nn as nn
inner.layers = nn.ModuleList(list(inner.layers[:43]))
if hasattr(model.config, "num_hidden_layers"): model.config.num_hidden_layers = 43
J = C.load_jlens(42, dev)
HOOK = {"h": None}
inner.layers[42].register_forward_hook(
    lambda m, i, o: HOOK.__setitem__("h", o[0] if isinstance(o, tuple) else o))
G = C.Grid(tok, C.TEMPLATES_RECOVERED[:NT], C.CARRIERS_RECOVERED[:NC], 42, J, dev)
NCELL = G.n_tpl * G.n_car
print("[grid] %dt x %dc = %d cells | sig %s | space=%s"
      % (G.n_tpl, G.n_car, NCELL, G.sig()[:10], SPACE), flush=True)

@torch.no_grad()
def read_cells(strings):
    """-> [B, NCELL, D] float32 on cpu. Buckets by token length: pre/post are fixed per cell and
    only the phrase slot varies, so equal-length phrases batch with NO padding inside the slot --
    padding there would shift the carrier positions we average over."""
    ids = {s: (tok(s, add_special_tokens=False).input_ids[:24] or
               tok(" the", add_special_tokens=False).input_ids) for s in set(strings)}
    pos = {s: k for k, s in enumerate(strings)}
    buckets = {}
    for s in strings: buckets.setdefault(len(ids[s]), []).append(s)
    DIM = J.shape[0]
    out = np.zeros((len(strings), NCELL, DIM), dtype="float32")
    ci = 0
    for c in range(G.n_car):
        for t in range(G.n_tpl):
            cell = G.cells[c][t]
            pre  = torch.tensor(cell["pre"],  device=dev)
            post = torch.tensor(cell["post"], device=dev)
            for _, grp in buckets.items():
                for a in range(0, len(grp), BATCH):
                    ch = grp[a:a+BATCH]
                    mid = torch.tensor([ids[s] for s in ch], device=dev)
                    B = mid.shape[0]
                    model(input_ids=torch.cat([pre.unsqueeze(0).expand(B,-1), mid,
                                               post.unsqueeze(0).expand(B,-1)], dim=1))
                    raw = HOOK["h"].float()[:, -cell["ncar"]:, :].mean(1)
                    # RAW L42, not J-space. reliability is defined as agreement between measured
                    # DIRECTIONS, and J is a fitted Jacobian rather than a rotation, so cos(Jx,Jy)
                    # != cos(x,y). J-space is right for the reward (whose NNOLS composition targets
                    # it) and wrong here. See results_jspectrum.json for the measured distortion.
                    v = (raw @ J.T if SPACE == "jspace" else raw).cpu().numpy()
                    for k, s in enumerate(ch): out[pos[s], ci] = v[k]
            ci += 1
    return out

# ---- per-cell means, computed ONCE and cached: what remains after subtraction is what
# distinguishes one span from another. Raw activations are ~93% a shared constant, so without this
# every span reads ~1.0 and the statistic is meaningless.
MU_PATH = "/vol/cellmean_%s_%dx%d_%s.npy" % (G.sig()[:10], G.n_tpl, G.n_car, SPACE)
if os.path.exists(MU_PATH):
    MU = np.load(MU_PATH); print("[mu] loaded %s" % MU_PATH, flush=True)
else:
    import random
    rng = random.Random(0)
    words = [l.strip() for l in open("/root/src/inv_core.py") if l.strip()][:0]  # placeholder
    files = sorted(glob.glob(IN + "/*.parquet"))
    fill = pq.read_table(files[0], columns=["span"]).to_pydict()["span"]
    fill = rng.sample(fill, 4096)
    MU = read_cells(fill).mean(axis=0)             # [NCELL, D]
    np.save(MU_PATH, MU)
    print("[mu] computed from 4096 fillers -> %s" % MU_PATH, flush=True)

JNP = None
MUJ = None


def stats_j(XJ):
    """Same statistic, in J-space. Uses its own per-cell mean (centering is space-specific)."""
    Z = XJ - MUJ[None, :, :]
    Z /= np.linalg.norm(Z, axis=-1, keepdims=True).clip(1e-8)
    n = Z.shape[1]
    def rho_of(U):
        m = U.mean(axis=1)
        V = 1.0 - (m * m).sum(-1)
        k = U.shape[1]
        return 1.0 - (k / (k - 1.0)) * V
    rho = rho_of(Z)
    loo = np.stack([rho_of(np.delete(Z, i, axis=1)) for i in range(n)], axis=1)
    sd = np.sqrt((n - 1.0) / n * ((loo - loo.mean(1, keepdims=True)) ** 2).sum(1))
    rc = np.clip(rho, 1e-6, 1 - 1e-6)
    return rho, rc / (1 - rc), sd


def stats(X):
    """X: [B, NCELL, D] raw cell reads -> per-span rho, S, jackknife CI, AND the atom vector.

    Reported in BOTH raw and J-space at no extra cost: the cell reads are raw, and J is a linear
    map, so the J-space statistic is a matmul away. Worth having both because they measure
    different things -- on 20 domain-labelled span sets, J-space separated concepts markedly better
    (d-prime 0.500 vs 0.313, AUC 0.634 vs 0.585) while the two agree only at r~0.88 pairwise.

    The stored VECTOR is raw L42: J can always be applied later, the reverse needs a pseudo-inverse.
    """
    Z = X - MU[None, :, :]
    Z /= np.linalg.norm(Z, axis=-1, keepdims=True).clip(1e-8)
    n = Z.shape[1]
    def rho_of(U):
        m = U.mean(axis=1)
        V = 1.0 - (m * m).sum(-1)
        k = U.shape[1]
        return 1.0 - (k / (k - 1.0)) * V
    rho = rho_of(Z)
    # leave-one-out jackknife over cells: no isotropy assumption, no extra forwards
    loo = np.stack([rho_of(np.delete(Z, i, axis=1)) for i in range(n)], axis=1)   # [B, n]
    sd = np.sqrt((n - 1.0) / n * ((loo - loo.mean(1, keepdims=True)) ** 2).sum(1))
    rho_c = np.clip(rho, 1e-6, 1 - 1e-6)
    return rho, rho_c / (1 - rho_c), sd

JNP = J.cpu().numpy().astype("float32")
MUJ_PATH = MU_PATH.replace(".npy", "_J.npy")
if os.path.exists(MUJ_PATH):
    MUJ = np.load(MUJ_PATH)
else:
    MUJ = np.einsum("cd,ed->ce", MU, JNP, optimize=True)
    np.save(MUJ_PATH, MUJ)
print("[mu] J-space cell means ready", flush=True)

files = sorted(glob.glob(IN + "/*.parquet"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT + "_vec", exist_ok=True)
spans, ntoks = [], []
for f in files:
    t = pq.read_table(f, columns=["span", "n_tokens"]).to_pydict()
    spans += t["span"]; ntoks += t["n_tokens"]
if LIMIT:
    # A smaller first dictionary is worth more than a bigger slower one: at the measured 196
    # cell-reads/s, 1M spans x 16 cells is ~2.8h on 8 GPUs versus ~1.2 days for 10M, so we learn
    # whether the artifact is actually useful before committing the long run. Deterministic prefix
    # of the shuffled bank, so a later 10M run is a strict superset.
    spans, ntoks = spans[:LIMIT], ntoks[:LIMIT]
    print("[limit] capped at %s spans" % "{:,}".format(len(spans)), flush=True)
lo = len(spans) * SHARD // NSHARD
hi = len(spans) * (SHARD + 1) // NSHARD
spans, ntoks = spans[lo:hi], ntoks[lo:hi]
print("[data] shard %d/%d -> %s spans" % (SHARD, NSHARD, "{:,}".format(len(spans))), flush=True)

# The rate/ETA clock must measure only work done THIS session. Previously t0 started before the
# skip loop while resumed chunks were credited to `done`, so a resumed shard charged ~1M spans of
# already-finished work against a few minutes of elapsed time: the printed cell-reads/s and eta
# were inflated by the resume fraction, and a run that had hours left looked nearly done.
t0 = time.time(); done = 0
fresh = 0          # spans actually measured in this process
t_fresh = None     # clock starts at the first non-skipped chunk
for k in range(0, len(spans), CHUNK):
    outp = "%s/rel-%05d-%08d.parquet" % (OUT, SHARD, k)
    if os.path.exists(outp):
        done += CHUNK; continue
    if t_fresh is None: t_fresh = time.time()
    sp = spans[k:k+CHUNK]; nt = ntoks[k:k+CHUNK]
    X = read_cells(sp)                                  # [B, NCELL, D] RAW
    rho, S, sd = stats(X)
    # the atom vector: mean over cells of the per-cell-centred reads, stored RAW in fp16
    VEC = (X - MU[None, :, :]).mean(axis=1).astype("float16")
    # same statistic in J-space (linear map, so no extra reads needed)
    XJ = np.einsum("bcd,ed->bce", X, JNP, optimize=True) if JNP is not None else None
    if XJ is not None:
        rho_j, S_j, sd_j = stats_j(XJ)
        del XJ
    else:
        rho_j = S_j = sd_j = np.full_like(rho, np.nan)
    del X
    pq.write_table(pa.table({
        "span": pa.array(sp, pa.string()),
        "n_tokens": pa.array(nt, pa.int16()),
        "rho": pa.array(rho.astype("float32")),
        "ci_lo": pa.array((rho - 1.96 * sd).astype("float32")),
        "ci_hi": pa.array((rho + 1.96 * sd).astype("float32")),
        "S": pa.array(S.astype("float32")),
        "n_cells": pa.array(np.full(len(sp), NCELL, dtype="int16")),
        "rho_j": pa.array(rho_j.astype("float32")),
        "ci_lo_j": pa.array((rho_j - 1.96*sd_j).astype("float32")),
        "ci_hi_j": pa.array((rho_j + 1.96*sd_j).astype("float32")),
        "S_j": pa.array(S_j.astype("float32")),
    }), outp, compression="zstd")
    # vectors in their own file so the metadata table stays small and previewable
    pq.write_table(pa.table({
        "span": pa.array(sp, pa.string()),
        "vector": pa.array(list(VEC), pa.list_(pa.float16(), 5120)),
    }), "%s_vec/vec-%05d-%08d.parquet" % (OUT, SHARD, k), compression="zstd")
    done += len(sp); fresh += len(sp)
    el = max(time.time() - (t_fresh if t_fresh is not None else t0), 1e-9)
    rate = fresh / el                                   # spans/s, this session only
    print("[prog] shard %d: %s/%s spans (%s new) | %.1f cell-reads/s | rho mean %.4f | eta %.1f h"
          % (SHARD, "{:,}".format(done), "{:,}".format(len(spans)), "{:,}".format(fresh),
             rate * NCELL, float(rho.mean()), (len(spans) - done) / max(rate, 1e-9) / 3600),
          flush=True)
print("RELIABILITY_SHARD_DONE %d" % SHARD, flush=True)
'''


# NOTE: 86400s is Modal's hard per-call ceiling. This pass takes ~27.5h for 11.6M spans,
# so it CANNOT finish in one call -- it is designed to be relaunched and resume from the
# chunks already on the volume. Do not change --nshard between runs: the resume key is
# the (shard, offset) filename, so a different shard count re-does everything.
@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=86400,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def measure(shard: int, nshard: int, nt: int = 4, nc: int = 4,
            in_dir: str = "/vol/spans_10m", out_dir: str = "/vol/reliability_10m",
            space: str = "raw", limit: int = 0):
    import subprocess
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")
    open("/root/w.py", "w").write(WORKER)
    rc = subprocess.run(["python", "/root/w.py"],
                        env=dict(os.environ, SHARD=str(shard), NSHARD=str(nshard),
                                 NT=str(nt), NC=str(nc), IN_DIR=in_dir, OUT_DIR=out_dir,
                                 SPACE=space, LIMIT=str(limit))).returncode
    VOL.commit()
    return rc


@app.local_entrypoint()
def smoke(nt: int = 4, nc: int = 4):
    """One 2000-span chunk on the FineFineWeb bank -- validates schema before the big run."""
    return measure.remote(0, 5000, nt, nc, "/vol/spans_ffw", "/vol/rel_smoke", "raw")


@app.local_entrypoint()
def main(nshard: int = 8, nt: int = 4, nc: int = 4, space: str = "raw",
         limit: int = 0, out: str = "/vol/rel_ffw_10m",
         in_dir: str = "/vol/spans_big_dedup"):
    # in_dir was hardcoded, so the vocabulary layer (/vol/spans_vocab) could not be measured
    # through this entrypoint at all. Default preserved so existing invocations are unchanged.
    rcs = list(measure.starmap([(i, nshard, nt, nc, in_dir, out, space, limit)
                                for i in range(nshard)]))
    print("return codes:", rcs)
