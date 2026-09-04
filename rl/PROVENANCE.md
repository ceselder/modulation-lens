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

---

## 2026-09-04 -- RL verdict: the 400-step ScaleRL run is a REGRESSION. Keep the SFT.

Run `modlens_scalerl400` (wandb rnghem5a): 8x B200 (2 rollout + 6 trainer), 16x256, ScaleRL recipe,
warm start `/vol/av_sft_4b/final`, frozen text->vector AR reward through J + affine. Stopped at
~step 260 of 400 once the evaluation below came back.

### The metric that matters

Reward is a learned surrogate, so it cannot certify a lens. `rl/diag_conditioning.py` measures, on
the 2048 holdout rows `build_bank.py` reserved, **greedily**:

    matched   reward(readout_i, target_i)
    permuted  reward(readout_i, target_{i+1})   and a random permutation
    delta     matched - permuted  =  the only part that required reading the activation

| checkpoint | matched | permuted | delta |
|---|---|---|---|
| SFT warm start | **0.7120 +- 0.0039** | 0.2352 | **0.4768** |
| RL step 50 | 0.4565 +- 0.0072 | 0.2204 | 0.2361 |
| RL step 200 | 0.4642 +- 0.0075 | 0.2303 | 0.2339 |

Matched fell 0.71 -> 0.46 (~35 SE); delta HALVED. Meanwhile the training reward rose 0.32 -> 0.85.
**The reward and the property it proxies moved in opposite directions for 200 steps.**

### Why

1. **A target-blind answer earns 0.343.** `rl/diag_constant_baseline.py`, 20k bank rows, targets =
   normalize(J.h - amu): mean pairwise cos 0.1182, best constant direction **0.3432**, variance in
   top 1/4/16 PCs 0.132/0.229/0.424. After whitening the same ceiling is **0.0064** (54x lower).
   The policy bought exactly this: from step 40 on, one of four bullets was a fixed phrase emitted
   verbatim for every activation ('* Spheres are unique in that every point on their surface is').
   By step 200 TWO of four were fixed even greedily, and matched 0.46 sits at the
   sqrt(0.229) = 0.48 fixed-4-basis ceiling.
2. **T=1.0 hides half the policy.** ScaleRL requires T=1.0 (the sampler's logprobs are the
   behaviour policy). Greedy SFT scores 0.712 where RL step 0 reported 0.321, so entropy collapse
   (3.66 -> 0.36) RECOVERS sampling loss and looks like learning. Never compare a T=1.0 reward
   curve against a greedy baseline.
3. **kl_coef 0.01 anchored nothing** -- kl_to_init 4.2 by step 80, asterisk spam at cos 0.000 past
   step ~170.

### Ruled out

* Injection is exact: `cos=1.0000 ratio=1.000 ||h||=24.0 (published 24.2) pre-marker max|d|=0`.
* The SFT does condition on the injection: delta 0.4768, ~60 SE. e.g. holdout row 0, whose text was
  "Are those famous beads around your neck cutting off the circulation to your brain?", reads out
  '* are you out of your mind / * the head of the company is a bit dim / ... / * the brain and the'.

### What a corrected run needs

* Remove the target-blind component: `--ar-whiten /vol/data/natural_whitener_jspace.npz
  --ar-whiten-key W_ridge0.1` (applied to BOTH sides -- the affine maps atoms into the UNWHITENED
  space), or `--reward-contrast-negatives N` (the permutation control as the objective).
* Far more KL than 0.01, and checkpoint selection on greedy holdout delta, never on reward.

### Dictionary status (same session)

`/vol/dict_5m` tier `all` VERIFIED complete: meta 8,028,555 rows == 33 vec shards, 8,028,555 rows
(<=12 tokens, newlines banned, rho mean 0.7015, 68 domains). Tier `f065` has a truncated shard from
a Modal preemption and must be regenerated -- it is derivable from `all` plus meta's `rho` column,
so the expensive measurement pass does NOT need repeating. Tiers 0.70/0.75 were never written.
`rl/verify_dict.py` re-runs the meta-vs-shard row-count check.

---

## 2026-09-04 (later) -- ROOT CAUSE FOUND: the reward was target-blind by construction

`rl_disagg.py` L2-normalised bank rows when building a rollout block (maemm only needs a DIRECTION
to steer), and those unit vectors were passed to `score(targets_are_raw=True)`. `target_space()`
computes `J.h - amu` where `amu` is a RAW-scale mean (raw ||h|| ~ 24, ||amu|| = 55.0). Subtracting a
55-norm mean from a 1-norm vector makes `-amu` dominate, so every target collapses onto ~the same
direction.

Reproduced exactly, inside the diagnostic harness, by scaling the targets and changing nothing else:

| targets | matched | permuted | delta | CONSTANT string |
|---|---|---|---|---|
| raw | 0.3346 | 0.1827 | 0.1520 | 0.0945 |
| **unit (what the trainer fed)** | **0.0637** | 0.0621 | **0.0016** | **0.1850** |
| trainer's own instrumentation | 0.0604 | 0.0589 | (r 0.002) | -- |

**The objective was inverted, not merely dead.** With unit targets a FIXED string scores 0.185
against a genuine readout's 0.064 -- 3:1 in favour of ignoring the activation. The 400-step run's
collapse onto one fixed bullet (then two) was the *optimal* response. It was never reward hacking,
and the earlier "basis padding" and "geometric reward is hackable" framings are downstream of this
one line.

**Why two runs missed it.** Both injection paths normalise internally (`_steer_vec`,
`make_inject_hook`), so the injected state was always right and rollouts stayed visibly on-topic --
bank row 4796 ("College at Rose Hill's 21st Annual Arts and Sciences Faculty Day") produced
"* President's College Lecturer Professor... * during a lecture on Modern European History...".
Correct readouts plus a meaningless reward looks like a model problem and is a plumbing problem.
`_verify_injection` cannot catch it: it injects a RANDOM unit vector and only checks the delta
equals `hnorm*STEER_COEFF*unit(v)`.

### Fix
* pass RAW rows at both block-builder sites (injection unaffected, reward repaired)
* `target_space()` now raises if median ||h|| < 0.10*||amu|| -- unit vectors rejected 5.5x over, raw
  accepted with 4.4x margin. Silence is what cost the day, so the assert stays.

### What this invalidates and what survives
* INVALID: every reward number from both RL runs, and the reward-based reasoning built on them
  (including my own "whitening is the fix" framing -- whitening addresses an exploit that only
  mattered because the real signal was absent).
* SURVIVES: all `diag_conditioning.py` numbers, which never route through the trainer -- SFT delta
  0.4768, RL step-50 0.2361, step-200 0.2339. "RL made the lens worse" still holds, now with a
  mechanism: it was trained against an inverted objective.
* STILL OPEN, separately: the rollout injects KARVONEN while the SFT and all evals use REPLACE,
  costing 34% of the delta (0.2298 -> 0.1520). Worth fixing; not the root cause.

---

## 2026-09-04 (post-fix) -- RL now IMPROVES the lens, in both injection modes

`modlens_fixed_contrast100` (wandb bdvkqi79), 8x B200, 16x256, contrast reward (1 negative,
group-strided), NO whitening, kl 0.1, raw targets. Greedy/sampled conditioning eval on 256 bank
rows, matched / permuted / delta:

| arm | matched | permuted | delta |
|---|---|---|---|
| SFT, replace/greedy (deployment mode) | 0.7120 | 0.2352 | 0.4768 |
| **step_10, replace/greedy** | **0.7323** | **0.2075** | **0.5248** |
| SFT, karvonen/T=1.0 (training mode) | 0.5786 | 0.2283 | 0.3503 |
| **step_10, karvonen/T=1.0** | **0.6527** | **0.2129** | **0.4398** |

delta +0.048 in the deployment mode (~6 SE on a delta SE of ~0.008) and +0.090 in the training
mode. matched rises AND permuted falls in both, which is the signature of better conditioning --
a shortcut would raise both together. The fixed constant string is unchanged at 0.1916.

Trainer-side decomposition over the same span: matched_fit 0.5522 -> 0.6212, neg_fit
0.2191 -> 0.2033. Reward 0.327 -> 0.417 with entropy 3.68 -> 3.42 and kl_to_init 0.16 -- compare the
broken run, which hit reward 0.738 with entropy 2.07 by step 20 because it was racing to a constant.

Trajectory in the deployment mode, before and after the normalisation fix:

    SFT baseline          0.4768
    broken run step 50    0.2361
    broken run step 200   0.2339
    FIXED run step 10     0.5248

Selection stays on this metric, never on reward.
