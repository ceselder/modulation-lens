# rl/ — vendored from ceselder/maemm, objective swapped

`rl.py`, `rl_disagg.py`, `modal_rl*.py`, `fast_lens_ext.py`, `rl_ddp.sh` and `mxf/` are copied
from **github.com/ceselder/maemm at commit bbddb76** (master, 2026-09-04). Their RL setup is
reused wholesale; only the objective is replaced.

## What was kept

`rl_disagg.py` is the fast path: X vLLM rollout GPUs + Y trainer GPUs. `--recipe scalerl` selects
the ScaleRL recipe (Khatri et al. 2025, arXiv 2510.13786) — PipelineRL with 8-step
off-policyness, interruption-based length control, FP32 logits (`install_fp32_head`), and
`J_ScaleRL` = prompt-level loss aggregation + batch-level advantage normalisation + CISPO
(truncated importance-sampling REINFORCE, lower clip fixed at 0) + zero-variance filtering +
no-positive resampling. No KL term. All of that is objective-agnostic and untouched.

Also kept, and worth knowing about: the trainer verifies injection **every step** via
`av/steer_apply_count` and masks rollouts whose output is CJK garbage — the signature of a failed
injection. Watch that before the reward.

## What was changed

One function. `R.score(texts, dirs_rep, actor, tok, device, a)` maps generated text to a reward;
maemm reads the text back through the clean base model and takes a position-max cosine against the
injected direction. Ours (`ar_reward.py`, opt-in via `--ar-reward`) splits the generation into
`--bullets` `'*'` lines, maps each to its modulation vector with the frozen text->vector AR,
combines them by **exact non-negative least squares**, and scores the cosine of that composition
against the target activation.

Flags added: `--ar-reward --ar-jlens --ar-affine --ar-amu --bullets --bullet-max-tok`.
Leave `--ar-reward` blank and the maemm objective runs unchanged.

## Facts the swap encodes (all measured, see diag_*.py)

* **The bank holds RAW L42 activations.** maemm's `dirs` are both the injected vector and the
  reward target; for us those differ, so the reward derives its own comparison space
  (`target_space`: J, minus the activation-pool mean, unit-norm). Pre-transforming the bank would
  inject the wrong thing *and* apply J twice.
* **The affine is required.** Atoms are modulation reads (a phrase in a template, pooled over
  carrier positions); targets are natural activations at one position — two different L42
  distributions. J alone gives 4-atom FVE 0.360, the affine alone 0.349, both 0.633. Do not
  whiten: whitening costs 6.3x.
* **The AR needs its OWN backbone, truncated to read_layer+1.** The identical adapter read through
  the full 64-layer stack HALVES the reward (0.331 vs 0.759 on identical atoms and activations).
  `attach()` refuses an untruncated backbone; `build_own()` loads a truncated one (~38 GB bf16).
* **Reward quality:** 0.759 against a 0.804 measured-vector reference = 94% retained;
  target-sensitive (right atoms with the wrong activation collapse to the 0.187 random floor);
  retention is flat ~95% at every arity k=1..12, so composition does not amplify AR error.
  Marginal cosine per bullet collapses (+0.085, +0.069, +0.053, +0.020 at k=2,4,8,12) -> k=4.
* **Do not select on the reward alone.** Three times in one session the geometric score and the
  readability moved independently. Judge checkpoints on readouts over the reserved holdout
  (`build_bank.py` keeps the last 2048 rows out of training for exactly this).
