"""How much reward does a TARGET-INDEPENDENT constant answer get?

The 400-step run collapsed onto one universal 4-bullet string scoring cos ~0.85 against arbitrary
activations. If the target distribution is anisotropic -- i.e. mean subtraction removes the first
moment but leaves one dominant principal direction -- then a constant is a near-optimal answer and
the reward carries almost no information about WHICH activation was injected. This measures that
directly, with no model in the loop:

  * mean pairwise cosine among targets (after J, minus AMU, normalised)
  * the best CONSTANT unit vector's mean cosine = the reward a target-blind policy can obtain
  * how much of that survives whitening
  * the same for k=4 non-negative combinations of fixed directions (the actual reward's arity)

If the best constant scores near what the collapsed policy scored, the objective as specified is
mostly satisfiable without reading the activation at all.
"""
import json, os, modal   # numpy is imported INSIDE run(): the local launcher env has no numpy

app = modal.App("modlens-diag-constant")
vol = modal.Volume.from_name("celeste-modlens-vol")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "numpy")
         .env({"HF_HOME": "/vol/.hf_home"})
         .add_local_file("rl/diag_constant_baseline.py", "/root/diag.py"))

JLENS = ("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/"
         "d740106d1e0f95456dc8718fba2895e9c8ffd6ef/qwen3.6-27b/j-lens/lens.pt")
AMU = "/vol/data/natural_whitener_jspace.npz"


@app.function(image=image, volumes={"/vol": vol}, gpu="A10G", timeout=3600, memory=65536)
def run(n: int = 20000, layer: int = 42):
    import numpy as np, torch
    dev = "cuda"
    J = torch.load(JLENS, map_location="cpu", weights_only=False)["J"][layer].to(dev).float()
    z = np.load(AMU)
    print("[amu] keys:", list(z.files), flush=True)
    amu = torch.tensor(z["mu"], device=dev).float()

    # Train-bank activations (what the RL actually optimised against) + the reserved holdout.
    out = {}
    for name, path in (("train_bank", "/vol/rl_bank/vecs.f32"),
                       ("holdout", "/vol/rl_bank/vecs_holdout.f32")):
        raw = np.fromfile(path, dtype="float32").reshape(-1, J.shape[0])
        raw = raw[:n]
        H = torch.from_numpy(raw).to(dev).float()
        Tt = H @ J.T - amu
        T = torch.nn.functional.normalize(Tt, dim=-1)

        # best constant unit vector = top eigenvector of T^T T; its mean cosine is the reward a
        # target-BLIND policy gets. (Also report the plain mean direction, which is what a
        # constant answer would naturally drift to.)
        mu_dir = torch.nn.functional.normalize(T.mean(0), dim=-1)
        cos_mu = (T @ mu_dir).mean().item()
        # power iteration for the top PC of the normalised targets
        v = torch.randn(T.shape[1], device=dev); v /= v.norm()
        for _ in range(200):
            v = T.T @ (T @ v); v /= v.norm()
        cos_pc = (T @ v).abs().mean().item()
        # spectrum: how concentrated is the target cloud?
        sv = torch.linalg.svdvals(T[: min(4096, T.shape[0])])
        ev = (sv ** 2) / (sv ** 2).sum()
        # mean pairwise cosine on a subsample
        sub = T[torch.randperm(T.shape[0], device=dev)[:2048]]
        G = sub @ sub.T
        off = G[~torch.eye(G.shape[0], dtype=torch.bool, device=dev)]
        # PRINT THE PRIMARY NUMBERS FIRST. The whitening branch below is the fragile part
        # (the mean-subtracted covariance is near-singular -- Cholesky failed at minor 4931/5120),
        # and on the first run it crashed before any of this reached stdout.
        print("\n=== %s (n=%d) ===" % (name, int(T.shape[0])), flush=True)
        print("  mean pairwise cos among targets : %.4f (sd %.4f)"
              % (float(off.mean()), float(off.std())), flush=True)
        print("  BEST CONSTANT answer, mean dir  : %.4f" % cos_mu, flush=True)
        print("  BEST CONSTANT answer, top PC    : %.4f   <-- reward obtainable WITHOUT reading the activation"
              % cos_pc, flush=True)
        print("  variance in top 1 / 4 / 16 PCs  : %.3f / %.3f / %.3f"
              % (float(ev[0]), float(ev[:4].sum()), float(ev[:16].sum())), flush=True)

        # whitened variant, for comparison. eigh + floor instead of cholesky(inv(C)): the
        # covariance is rank-deficient, so the direct inverse is not positive-definite.
        cos_pc_w = float("nan")
        try:
            Xc = Tt - Tt.mean(0)
            C = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
            w, Q = torch.linalg.eigh(C.double())
            w = torch.clamp(w, min=1e-6 * float(w.max()))       # floor the near-null directions
            W = (Q * w.rsqrt()) @ Q.T
            Tw = torch.nn.functional.normalize((Xc.double() @ W).float(), dim=-1)
            vw = torch.randn(Tw.shape[1], device=dev); vw /= vw.norm()
            for _ in range(200):
                vw = Tw.T @ (Tw @ vw); vw /= vw.norm()
            cos_pc_w = (Tw @ vw).abs().mean().item()
            subw = Tw[torch.randperm(Tw.shape[0], device=dev)[:2048]]
            Gw = subw @ subw.T
            offw = Gw[~torch.eye(Gw.shape[0], dtype=torch.bool, device=dev)]
            print("  --- after whitening ---", flush=True)
            print("  mean pairwise cos among targets : %.4f" % float(offw.mean()), flush=True)
            print("  BEST CONSTANT answer, top PC    : %.4f" % cos_pc_w, flush=True)
        except Exception as e:
            print("  [whitening failed] %s" % str(e)[:160], flush=True)

        r = {"n": int(T.shape[0]),
             "mean_pairwise_cos": float(off.mean()), "sd_pairwise_cos": float(off.std()),
             "best_constant_cos_meandir": cos_mu, "best_constant_cos_toppc": cos_pc,
             "top1_var_frac": float(ev[0]), "top4_var_frac": float(ev[:4].sum()),
             "top16_var_frac": float(ev[:16].sum()),
             "best_constant_cos_toppc_WHITENED": cos_pc_w}
        out[name] = r
    os.makedirs("/vol/diag", exist_ok=True)
    json.dump(out, open("/vol/diag/constant_baseline.json", "w"), indent=1)
    vol.commit()
    return out


@app.local_entrypoint()
def main(n: int = 20000):
    print(json.dumps(run.remote(n=n), indent=1))
