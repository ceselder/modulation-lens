"""Does the LoRA adapter's weight norm grow, and does it track the gradient norm?

Pre-clip gradient norm rose 0.55 -> 3.2 (5.8x) over 350 steps while normalised advantages have unit
variance by construction and generated length grew only 27%. The remaining candidate is LoRA's
factored parameterisation: with z = W0 x + (alpha/sqrt(r)) B A x,

    grad_A  ~  B^T (dL/dz) x^T      ->  ||grad_A|| scales with ||B||
    grad_B  ~  (dL/dz) (A x)^T      ->  ||grad_B|| scales with ||A||

so the gradient w.r.t. each factor scales with the OTHER factor's norm, and with kl_beta 0, no weight
decay and no importance ratio there is nothing bounding weight growth. Prediction: the product
||A||*||B|| grows by roughly the same factor as the gradient norm.

Runs on CPU against the volume -- no 1.28 GB downloads.
"""
import os
import re

import modal

app = modal.App("celeste-modlens-norms")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = modal.Image.debian_slim(python_version="3.12").pip_install("safetensors", "torch==2.8.0")


@app.function(image=img, volumes={"/vol": VOL}, timeout=1800, cpu=4.0)
def norms(run_dir: str = "/vol/ckpts/rl_modal_g16_t8_m39"):
    import torch
    from safetensors.torch import load_file
    out, ref = [], [None]
    steps = sorted(d for d in os.listdir(run_dir) if re.fullmatch(r"iter_\d+", d))
    for d in steps:
        p = os.path.join(run_dir, d, "adapter_model.safetensors")
        if not os.path.exists(p):
            continue
        sd = load_file(p)
        na = nb = 0.0
        for k, v in sd.items():
            f = v.float()
            if "lora_A" in k:
                na += float((f ** 2).sum())
            elif "lora_B" in k:
                nb += float((f ** 2).sum())
        na, nb = na ** 0.5, nb ** 0.5
        # DISPLACEMENT from the first checkpoint, not just the norm: ||A|| is preserved under
        # rotation, so a stable norm does NOT mean the weights are static. This is the quantity
        # that says whether the adapter actually moved.
        if ref[0] is None:
            ref[0] = {k: v.float().clone() for k, v in sd.items()}
            dsp = rel = 0.0
        else:
            num = den = 0.0
            for k, v in sd.items():
                if k in ref[0]:
                    num += float(((v.float() - ref[0][k]) ** 2).sum())
                    den += float((ref[0][k] ** 2).sum())
            dsp, rel = num ** 0.5, (num ** 0.5) / max(den ** 0.5, 1e-9)
        out.append({"step": int(d.split("_")[1]), "normA": na, "normB": nb,
                    "displacement": dsp, "relative_displacement": rel})
        print("%-14s ||A|| %8.3f ||B|| %8.3f  ||W(t)-W(0)|| %9.4f  relative %7.4f"
              % (d, na, nb, dsp, rel), flush=True)
    if len(out) > 1:
        print("\nrelative displacement from the first checkpoint: %.4f  (%.2f%% of ||W(0)||)"
              % (out[-1]["relative_displacement"], 100 * out[-1]["relative_displacement"]),
              flush=True)
    return out
