"""Modal app: ScaleRL for the MODULATION LENS -- maemm's disaggregated trainer, our objective.

Vendored setup (rl/PROVENANCE.md): rl_disagg.py is maemm's fast path, X vLLM rollout GPUs + Y HF
trainer GPUs in one container. `--recipe scalerl` selects PipelineRL-8 lag, CISPO, prompt-level
aggregation, batch advantage normalisation, zero-variance filtering, NPR and FP32 logits -- all
objective-agnostic. The objective is swapped by --ar-reward (rl/ar_reward.py).

Objective: the policy is injected with a real L42 activation and writes --bullets '*' lines; each
bullet is mapped to its modulation vector by the FROZEN text->vector AR; they are combined by
exact non-negative least squares; the reward is the cosine of that composition with the
activation. Validated at 94% of the measured-vector reference, target-sensitive (shuffled targets
collapse to the random floor), ~95% retention at every arity 1..12.

Image pins follow maemm exactly (vllm 0.19 + transformers 5.15), which is what works for
Qwen3.6-27B there. NOTE: the AR adapter was TRAINED under transformers 5.5.4, and a
transformers-version difference is what produced the 0.331-vs-0.759 backbone surprise -- so run
`modal run modal_rl_modlens.py::calibrate` in THIS image before trusting the reward here.

Launch (MODAL_PROFILE=safety-sahan):
    modal run modal_rl_modlens.py::calibrate                                    # reward sanity IN this image
    DISAGG_GPU=B200:4 modal run --detach modal_rl_modlens.py::main --n-rollout 1 --n-trainer 3 --total-steps 6
    DISAGG_GPU=B200:8 modal run --detach modal_rl_modlens.py::main --n-rollout 2 --n-trainer 6 \
        --total-steps 400 --extra-args "--run-name modlens_scalerl_16x256"
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
app = modal.App("modlens-rl-scalerl")
GPU = os.environ.get("DISAGG_GPU", "B200:4")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
    .pip_install("transformers==5.15.0", "peft==0.20.0", "accelerate==1.14.0",
                 "wandb==0.28.2", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet", "pyarrow")
    .pip_install("flash-linear-attention==0.5.2")
    .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(REPO / "rl" / "rl.py", "/pmx/RL/rl_hf.py")
    .add_local_file(REPO / "rl" / "rl_disagg.py", "/pmx/RL/rl_disagg.py")
    .add_local_file(REPO / "rl" / "ar_reward.py", "/pmx/RL/ar_reward.py")     # the swapped objective
    .add_local_file(REPO / "rl" / "fast_lens_ext.py", "/pmx/helpers/fast_lens_ext.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

# one volume: bank, AR adapter, affine, whitener and the HF cache all live on it
vol = modal.Volume.from_name("celeste-modlens-vol")

BANK_DIR = "/vol/rl_bank"                       # vecs.f32 (497,952 raw L42 activations) + holdout
# FROZEN reward model, not the live training path. /vol/ar_l42_text2vec is overwritten on every
# held-out improvement, so pointing the reward at it would make the run non-reproducible: the
# calibration and the training steps would see different AR weights. ar_frozen_rl_v1 is the
# step-1300 snapshot (held-out cos 0.9280, ridge reference 0.6527) and refuses overwrite.
AR_DIR = "/vol/ar_frozen_rl_v1"
AFFINE = "/vol/data/affine_M_jspace.npy"        # the 1.76x atom->activation alignment
AMU = "/vol/data/natural_whitener_jspace.npz"   # 'mu': the activation-pool mean, subtracted in J
JLENS_GLOB = ("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/*/"
              "qwen3.6-27b/j-lens/lens.pt")
CKPT_DIR = "/vol/ckpts_modlens_scalerl"
AV_SFT = "/vol/av_sft_4b"                       # the 4-bullet SFT warm start (adapter + prompt.txt)

# 16 x 256 (user). maemm production is 8 x 128; this is 4x the rollout volume per step, which the
# frozen AR makes affordable: it replaces 16 grid forwards per bullet with one.
TRAIN_ARGS = [
    "--data-dir", BANK_DIR,
    "--bank-file", "vecs.f32",
    "--recipe", "scalerl",
    "--group-size", "16",
    "--groups-per-step", "256",
    # ---- the modulation-lens objective ----
    "--ar-reward", AR_DIR,
    "--ar-affine", AFFINE,
    "--ar-amu", AMU,
    "--bullets", "4",                # marginal cosine per bullet collapses past 4 (+0.069 -> +0.020)
    "--bullet-max-tok", "12",        # matches the dictionary's <=12-token cap
    # A lens must be READ with the prompt it was TRAINED on. The AV was SFT'd on inv_train's job
    # prompt (marker at 40 of 186), not maemm's (marker last), so the warm start only transfers
    # with this override. maemm argues a mid-prompt marker weakens conditioning -- worth testing
    # against their layout once this run has a baseline.
    "--prompt-file", AV_SFT + "/prompt.txt",
    "--reward-metric", "cosine",
    "--reward-scale", "1",
    # ---- what the first two runs actually needed ----
    # ROOT CAUSE (fixed in rl_disagg): bank rows were L2-normalised for steering and then used as
    # reward targets, so subtracting the RAW-scale amu (||amu||=55) collapsed every target onto
    # -amu. A FIXED string scored 0.185 against a real readout's 0.064 -- the objective REWARDED
    # ignoring the activation 3:1, which is why the policy collapsed onto fixed bullets.
    #
    # With that fixed, WHITENING IS NO LONGER THE RIGHT TOOL and is deliberately off. Its only job
    # was crushing the target-blind ceiling (0.343 -> 0.0064), but it halves the signal too
    # (delta 0.4768 -> 0.2617 unwhitened->whitened, replace/greedy). The contrastive term removes
    # the target-blind component BY CONSTRUCTION and at full strength: a constant answer scores
    # fit(const,t_i) - fit(const,t_j) ~= 0 regardless of how much shared structure the space has.
    "--reward-contrast-negatives", "1",
    "--reward-contrast-weight", "1.0",
    # LEGIBILITY PRESSURE -- calibrated, after getting this wrong in both directions.
    #
    # I first claimed a 6-step ungated run reward-hacked (reward 0.32 -> 0.43 with "degenerate"
    # rollouts). That was WRONG: it came from the stdout `sample` line, which truncates at ~100
    # chars, and the two fragments I quoted were the 4th bullet of rows whose first three bullets
    # were fluent English. Reading all 16 rows/step, text quality IMPROVED as reward rose
    # (non-ASCII 0.022 -> 0.002; step-0 junk like '* yeratarau te wair' gone by step 5).
    #
    # The gates I then added were miscalibrated the other way: --fluency-floor -4.0 rejected 99.4%
    # of rollouts (reward/gate_frac 0.0056). That logp is UNCONDITIONED (raw text, no prompt), so
    # bullet fragments sit far below conditioned prose; -4.0 was a guess and a bad one. A gate that
    # fires on ~all samples is a near-constant offset that cancels under batch-normalized
    # advantages, adds variance, and hands the surviving 0.6% outsized advantage.
    #
    # So: MEASURE first (--flu-monitor-every logs logp/distinct percentiles without gating), keep
    # the one gate that was actually validated, and re-add a fluency floor only from percentiles.
    "--no-fluency-floor",            # was -4.0: rejected 99.4% of legible rollouts. Measure first.
    "--distinct-floor", "0.6",       # VALIDATED: min 0.706 over 32 good rollouts, so non-binding
                                     # on good text but still catches repetition spam. Free (no forward).
    "--gate-penalty", "0.5",
    "--flu-monitor-every", "10",
    "--flu-monitor-n", "256",
    # ScaleRL drops KL because its rewards are VERIFIABLE; ours is a learned surrogate and
    # therefore hackable in principle, so keep a small anchor to the warm start. Cheap insurance:
    # measured kl 0.0003-0.0010, i.e. it is not currently binding. Explicit flags beat the bundle.
    # NO KL. It was added when the reward was inverted and the policy raced to a constant
    # (kl_to_init 4.2 by step 80) -- but that was the broken objective, not a property of this one.
    # With the reward fixed there is no degeneration to anchor against: 16/16 unique rollout texts at
    # every logged step and ZERO first bullets shared across activations, over 100 steps. Meanwhile
    # the anchor holds the policy near the SFT (kl_to_init ~0.13, gnorm 0.17 vs clip 1.0), which is
    # the opposite of what a run that has barely moved needs. ScaleRL drops KL by design.
    "--kl-coef", "0",
    # ---- generation ----
    "--min-new-tokens", "16",
    "--max-new-tokens", "96",        # 4 bullets x <=12 tokens + '* ' scaffolding
    "--len-penalty-start", "8",
    "--len-penalty-per-tok", "0.00025",
    # ---- optimisation ----
    # 1e-5. The 3e-5 default in this project is for RECONSTRUCTION/injection training (the AR and
    # the AV), not RL -- misapplying it there was my error; for RL 1e-4 ran hot with a decay
    # signature and 1e-5 is the setting.
    "--lr", "1e-5",
    "--warmup-steps", "10",
    "--max-grad-norm", "1",
    "--adam-eps", "1e-8",
    "--adam-betas", "0.9", "0.95",
    "--score-batch", "128",
    "--vllm-gpu-mem", "0.85",
    # CAP the concurrent sequences per rollout rank. --max-num-seqs defaults to
    # rollout_block_groups * group_size, which at 16 x 256 is 4096 -- and vLLM's GDN
    # linear-attention state buffer scales with it: gdn_linear_attn.py tried to allocate 18.09 GiB
    # at engine start and OOMed the rollout rank. maemm's production is 1024 seqs over 2 rollout
    # GPUs = 512/GPU, which is proven on B200 (~2.3 GB for the same buffer). The step's 4096
    # rollouts are then generated in waves, which the disaggregated design overlaps with training.
    "--max-num-seqs", "512",
    "--save-every", "0",
    # The regression was ALREADY complete by step 50 last time (greedy delta 0.477 -> 0.236), so
    # 400 steps just burned GPU past the point of failure. Short run, dense early checkpoints, and
    # selection on greedy holdout delta via rl/diag_conditioning.py -- never on reward.
    "--save-steps", "10,25,50,100",
    "--save-dir", CKPT_DIR,
]


def _jlens():
    import glob
    p = glob.glob(JLENS_GLOB)
    if not p:
        raise SystemExit("j-lens not found at %s" % JLENS_GLOB)
    return p[0]


@app.function(image=image, volumes={"/vol": vol}, gpu="B200", timeout=7200,
              secrets=[modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})])
def calibrate(n: int = 256, k: int = 4):
    """Re-run the reward calibration INSIDE this image before trusting it here.

    The AR was trained under transformers 5.5.4; this image pins 5.15. A transformers difference is
    exactly what produced the 0.331-vs-0.759 truncation surprise, so verify rather than assume:
    true atoms should beat random atoms by a wide margin AND beat shuffled targets by the same
    margin, at ~94% of the measured-vector reference.
    """
    import json, sys
    import numpy as np, torch
    import torch.nn.functional as F
    sys.path.insert(0, "/pmx/RL")
    import ar_reward as ARR
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    R = ARR.ARReward(AR_DIR, _jlens(), AFFINE, device="cuda", read_layer=42,
                     max_tokens=12, amu_path=AMU)
    R.build_own("Qwen/Qwen3.6-27B")

    rows = []
    with open("/vol/data/nnomp_4bullets_sft.jsonl") as f:
        for i, l in enumerate(f):
            if i >= n: break
            rows.append(json.loads(l))
    bank = np.memmap(f"{BANK_DIR}/vecs.f32", dtype=np.float32, mode="r").reshape(-1, 5120)
    ACT = torch.from_numpy(np.stack([bank[r["i"]] for r in rows]).copy()).cuda()

    import random
    rng = random.Random(0)
    pool = [b for r in rows for b in r["bullets"]]
    per = [r["bullets"][:k] for r in rows]
    perm = list(range(len(rows))); rng.shuffle(perm)
    ref = float(np.mean([r["fve"] for r in rows])) ** 0.5
    out = {}
    for name, pre, tgt in (("true", per, ACT),
                           ("random", [rng.sample(pool, k) for _ in rows], ACT),
                           ("shuffled", per, ACT[torch.tensor(perm)])):
        r = R.score([""] * len(rows), tgt, None, tok, k=k, max_tok=12, pre_split=pre)
        out[name] = float(r.mean())
        print("  %-10s cos %.4f" % (name, out[name]), flush=True)
    print("\n[ref] measured-vector reference cos %.4f" % ref, flush=True)
    retained = (out["true"] - out["random"]) / max(ref - out["random"], 1e-9)
    print("[retained] %.1f%% of the signal above the random floor" % (100 * retained), flush=True)
    ok = (out["true"] > out["random"] + 0.1 and out["true"] > out["shuffled"] + 0.1
          and retained > 0.5)
    print("[verdict] %s" % ("PASS" if ok else "FAIL -- do NOT launch the RL on this reward"),
          flush=True)
    json.dump(out | {"ref": ref, "retained": retained, "ok": bool(ok)},
              open("/vol/data/rl_image_calibration.json", "w"), indent=1)
    vol.commit()
    return out


@app.function(image=image, volumes={"/vol": vol}, gpu=GPU, timeout=24 * 3600,
              secrets=[modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})])
def train(n_rollout: int = 1, n_trainer: int = 3, total_steps: int = 6, extra_args: str = "",
          init_adapter: str = "", no_wandb: bool = False):
    import subprocess, sys
    args = list(TRAIN_ARGS) + ["--ar-jlens", _jlens(), "--total-steps", str(total_steps)]
    if init_adapter:
        args += ["--init-adapter", init_adapter, "--ref-adapter", init_adapter]
    if no_wandb:
        args += ["--no-wandb"]
    if extra_args:
        args += extra_args.split()
    # --role MUST be present in argv even though it defaults to "launch": the launcher rewrites
    # its own argv for the children (child_argv[child_argv.index("--role") + 1] = role), so a
    # missing flag raises ValueError("'--role' is not in list") before anything starts.
    # cwd=/pmx with a RELATIVE script path, matching maemm's launcher -- the rollout/trainer
    # children resolve RL/ and helpers/ from there.
    cmd = ["python", "RL/rl_disagg.py", "--role", "launch",
           "--n-rollout", str(n_rollout), "--n-trainer", str(n_trainer)] + args
    print("[launch] %s" % " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONPATH="/pmx/RL:/pmx/helpers:/pmx/eval")
    return subprocess.run(cmd, cwd="/pmx", env=env).returncode


@app.local_entrypoint()
def main(n_rollout: int = 1, n_trainer: int = 3, total_steps: int = 6, extra_args: str = "",
         init_adapter: str = "", no_wandb: bool = False):
    print("rc =", train.remote(n_rollout=n_rollout, n_trainer=n_trainer, total_steps=total_steps,
                               extra_args=extra_args, init_adapter=init_adapter,
                               no_wandb=no_wandb))
