"""How much does the J-lens transform distort cosine similarity?

A linear map preserves cosines only if it is orthogonal (J^T J = cI). J is a FITTED Jacobian, so it
is not, and the reliability statistic -- defined as agreement between measured directions -- is
therefore NOT the same quantity in raw L42 space and in J-space. This matters because the harvest
was about to compute it in J-space (inv_core's Grid.read multiplies by J.T, because that is what the
REWARD needs) while thinkies-v3 defines it on raw L42.

Quantifies the distortion three ways:
  1. singular-value spectrum of J: condition number, and the participation ratio (how many
     directions carry the action). A near-orthogonal J would have a flat spectrum.
  2. direct empirical test: sample random pairs of REAL atom vectors and compare cos(x,y) against
     cos(Jx, Jy). The scatter and the mean shift are the answer to "does this change things".
  3. the same comparison restricted to NEARBY pairs (cos > 0.3), which is the regime the reliability
     statistic actually lives in -- 16 draws of the same phrase are all near each other, and a
     distortion that only shows up for orthogonal pairs would be irrelevant here.
"""
import os

import modal

app = modal.App("celeste-jlens-spectrum")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("torch==2.8.0", "numpy", "pyarrow", "huggingface_hub")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1"})
       .add_local_dir("/home/celeste/modlens_modal/src", "/root/src", copy=True))

WORKER = r'''
import glob, json, os, sys
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, "/root/src")
import inv_core as C
dev = "cuda"
J = C.load_jlens(42, dev)
print("[J] shape %s dtype %s" % (tuple(J.shape), J.dtype), flush=True)

s = torch.linalg.svdvals(J.float())
s = s.cpu().numpy()
pr = float(s.sum()**2 / (s**2).sum())
print("\n[1] SINGULAR VALUES of J")
print("    max %.4f  min %.6f  condition number %.1f" % (s.max(), s.min(), s.max()/max(s.min(),1e-12)))
print("    participation ratio %.1f of %d dims (flat/orthogonal would be %d)" % (pr, len(s), len(s)))
for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
    print("      p%-3d %.4f" % (q, np.percentile(s, q)))
print("    ratio p95/p5 = %.2f   (1.0 = perfectly orthogonal, no cosine distortion)"
      % (np.percentile(s,95)/max(np.percentile(s,5),1e-12)))

# real atom vectors
labs, vecs = [], []
for sh in sorted(glob.glob("/vol/thinkies_v3/thinkies_v3-*-of-00007.parquet")):
    for b in pq.ParquetFile(sh).iter_batches(batch_size=16384, columns=["label","vector"]):
        labs += b.column("label").to_pylist()
        vecs.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                               dtype="float32").reshape(-1, 5120))
        if sum(v.shape[0] for v in vecs) >= 40000: break
    break
A = torch.from_numpy(np.concatenate(vecs)[:40000]).to(dev)
AJ = A @ J.T
An  = A  / A.norm(dim=1, keepdim=True).clamp(min=1e-8)
AJn = AJ / AJ.norm(dim=1, keepdim=True).clamp(min=1e-8)
print("\n[2] EMPIRICAL: cos(x,y) vs cos(Jx,Jy) on %d real atom vectors" % A.shape[0])
g = torch.Generator(device="cpu").manual_seed(0)
i = torch.randint(0, A.shape[0], (200000,), generator=g).to(dev)
j = torch.randint(0, A.shape[0], (200000,), generator=g).to(dev)
keep = i != j
i, j = i[keep], j[keep]
c_raw = (An[i] * An[j]).sum(1)
c_j   = (AJn[i] * AJn[j]).sum(1)
def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a*b).sum()/torch.sqrt((a*a).sum()*(b*b).sum()))
print("    all pairs   : raw mean %.4f (sd %.4f) | J mean %.4f (sd %.4f) | r = %.4f"
      % (c_raw.mean(), c_raw.std(), c_j.mean(), c_j.std(), pear(c_raw, c_j)))
print("    mean |cos_J - cos_raw| = %.4f" % (c_j - c_raw).abs().mean())

print("\n[3] NEARBY pairs only -- the regime the reliability statistic lives in")
for thr in (0.2, 0.3, 0.5):
    m = c_raw > thr
    if int(m.sum()) < 500: continue
    print("    cos_raw > %.1f (n=%7d): raw mean %.4f | J mean %.4f | r = %.4f | mean shift %+.4f"
          % (thr, int(m.sum()), c_raw[m].mean(), c_j[m].mean(), pear(c_raw[m], c_j[m]),
             (c_j[m]-c_raw[m]).mean()))

json.dump({"cond": float(s.max()/max(s.min(),1e-12)), "pr": pr,
           "p95_over_p5": float(np.percentile(s,95)/max(np.percentile(s,5),1e-12)),
           "r_all": pear(c_raw, c_j), "mean_abs_shift": float((c_j-c_raw).abs().mean()),
           "raw_mean": float(c_raw.mean()), "j_mean": float(c_j.mean())},
          open("/vol/results_jspectrum.json","w"), indent=1)
print("\nJSPEC_DONE", flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=3600)
def run():
    import subprocess
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")
    open("/root/w.py", "w").write(WORKER)
    rc = subprocess.run(["python", "/root/w.py"]).returncode
    VOL.commit()
    return rc
