#!/usr/bin/env python3
"""
Whitener fitted on NATURAL-TEXT layer-42 activations, matching build_probe_whitener.py's math
exactly but on the right distribution.

Why: the existing probe_whitener.npz was fitted on av_L42_150k (the AV probe distribution), and its
own docstring insists mu and Sigma must come from "the activations being reconstructed". When the
thing being read is a natural-prose position, that is the wrong distribution -- measured: the probe
mean has norm 65.07 while natural-prose L42 positions give 78.7. Using the mismatched whitener
leaves a constant the atoms can absorb, which is the loophole whitening was supposed to close.

Same construction as the original so the two are comparable:
    mu     = mean over positions
    Sigma  = covariance of (X - mu)
    lam    = ridge_frac * mean(eigenvalue)
    W      = (Sigma + lam I)^(-1/2)   via eigendecomposition
"""
import os
import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.6-27B"
LAYER = int(os.environ.get("NW_LAYER", "42"))
N = int(os.environ.get("NW_N", "60000"))
RIDGES = [float(x) for x in os.environ.get("NW_RIDGES", "0.01,0.1").split(",")]
CORPUS = os.environ.get("NW_CORPUS", "/workspace/.hf_home/hub/datasets--HuggingFaceFW--"
                        "fineweb-edu/snapshots/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/"
                        "sample/10BT/000_00000.parquet")
OUT = os.environ.get("NW_OUT", "/workspace/cnla/skip-lens/data/meansub/natural_whitener.npz")
JT = int(os.environ.get("NW_JTRANSFORM", "0"))     # 1 = fit in J-transformed (layer-62) space

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to("cuda").eval()
grab = {}
model.model.layers[LAYER].register_forward_hook(
    lambda m, i, o: grab.__setitem__("h", o[0] if isinstance(o, tuple) else o))

J = None
if JT:
    import glob
    lp = glob.glob("/workspace/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/*/"
                   "qwen3.6-27b/j-lens/lens.pt")[0]
    J = torch.load(lp, map_location="cpu", weights_only=False)["J"][LAYER].to("cuda").float()
    print("[nw] fitting in J-transformed (layer-62) space", flush=True)

print("[nw] harvesting %d positions from %s" % (N, os.path.basename(CORPUS)), flush=True)
rows, got = [], 0
pf = pq.ParquetFile(CORPUS)
with torch.no_grad():
    for b in pf.iter_batches(batch_size=64, columns=["text"]):
        for t in b.to_pydict()["text"]:
            if not t or len(t) < 400:
                continue
            ids = tok(t, add_special_tokens=False, truncation=True, max_length=256).input_ids
            if len(ids) < 32:
                continue
            model(input_ids=torch.tensor([ids], device="cuda"))
            H = grab["h"].float()[0]
            if J is not None:
                H = H @ J.T
            # skip the first few positions: they carry position-specific structure rather than
            # the content distribution we want to whiten against
            rows.append(H[4:].cpu())
            got += H.shape[0] - 4
            if got >= N:
                break
        if got >= N:
            break
        if len(rows) % 200 == 0:
            print("  %d positions" % got, flush=True)

X = torch.cat(rows)[:N].to("cuda")
n, d = X.shape
print("[nw] X %s" % (tuple(X.shape),), flush=True)

mu = X.mean(0)
Xc = X - mu
print("[nw] |mu|=%.2f  mean |h-mu|=%.2f" % (float(mu.norm()), float(Xc.norm(dim=1).mean())),
      flush=True)

Sigma = (Xc.T @ Xc) / (n - 1)
w, U = torch.linalg.eigh(Sigma.double())
w = w.clamp(min=0)
print("[nw] Sigma eig: min %.4g max %.4g mean %.4g cond %.3g"
      % (float(w.min()), float(w.max()), float(w.mean()),
         float(w.max() / w.clamp(min=1e-12).min())), flush=True)
c = torch.cumsum(w.flip(0), 0) / w.sum()
k90 = int((c < 0.9).sum()) + 1
print("[nw] top-%d of %d eigendirections hold 90%% of the variance" % (k90, d), flush=True)

out = {"mu": mu.cpu().numpy(), "eigvals": w.float().cpu().numpy(), "n": np.int64(n),
       "k90": np.int64(k90), "layer": np.int64(LAYER), "jtransformed": np.int64(JT)}
for rf in RIDGES:
    lam = rf * float(w.mean())
    Wm = (U * ((w + lam) ** -0.5).unsqueeze(0)) @ U.T
    out["W_ridge%s" % rf] = Wm.float().cpu().numpy()
    out["lam_ridge%s" % rf] = np.float32(lam)
    print("[nw] ridge %.3g -> lam %.4g, |W|_F %.2f" % (rf, lam, float(Wm.norm())), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **out)
print("[nw] wrote %s (%.0f MB)" % (OUT, os.path.getsize(OUT) / 1e6), flush=True)
print("NATURAL_WHITENER_DONE", flush=True)
