"""Disaggregated GRPO for the MAEMM universal inverter: X vLLM ROLLOUT GPUs + Y HF TRAINER GPUs in ONE
container (N = X + Y processes, one GPU each), coupled through the container-local filesystem.

Why. rl/rl.py hosts BOTH the HF actor (61 GB resident, ~98 GB peak) AND a vLLM engine on EVERY GPU:
micro-batch is stuck at 3, vLLM gets a third of the GPU, old_logp is recomputed in HF, and the vllm_lens
hook is O(layers x reqs x keys) per decode step -> 145-250 s per 1024-rollout step on 4 GPUs. Here every
GPU does ONE job, the trainer never runs vLLM (-> no grad checkpointing, micro-batch 16-32), the sampler's
own per-token logprobs ARE old_logp (no HF recompute; the 1-2 step policy lag is importance-corrected by
the PPO clip + TIS cap from rl.py), and rollout GPUs run vLLM at 0.85 memory / 1024 seqs with a fast
steering hook (rl/fast_lens_ext.py) and optional CUDA graphs for decode.

Roles (this file, selected by --role; the launcher spawns the others):
  launch   parent: N children, CUDA_VISIBLE_DEVICES=<one gpu> each (GPUs [0,Y) trainers, [Y,N) rollouts),
           streams their stdout with [T<k>]/[R<r>] prefixes, tears everything down when the trainer
           finishes or anything dies.
  trainer  DDP group of Y ranks (gloo or nccl) over the HF actor + LoRA + AdamW (+ frozen 'ref' adapter):
           consume rollout blocks -> reward with rl.py's score() on the clean base -> compute_advantages
           -> clipped policy gradient with vLLM logprobs as old_lp -> capped-k3 KL to init -> exact
           token/seq-weighted grad all-reduce -> AdamW -> rank 0 publishes the adapter (+ ||h_marker||)
           for the rollout ranks every --publish-every steps, checkpoints every --save-every, wandb.
  rollout  one vLLM engine (TP=1, LoRA, full GPU); loop: pick up the newest published adapter, draw a
           block of directions from the bank, generate G samples each with per-request steering at the
           marker, write a rollout block file; block until the queue drains below --max-queue-blocks.

Filesystem protocol (all under --work-dir, default /tmp/disagg):
  lora/step_<k>/{adapter_model.safetensors, adapter_config.json, meta.json}   vLLM key layout
      (rl.py _save_adapter_for_vllm) + meta {"step", "hnorm": ||h_marker|| with the adapter ON}
  lora/latest            the integer k, written atomically (tmp + os.replace); rollouts reload on change
  queue/blk_<adapter_step>_<t_ns>_<rank>.pt   torch.save dict: dir_idx, dirs [B_r,d], gen_ids, lps
      (vLLM per-token logprobs; None where the engine dropped the stop token -> ratio 1 there),
      adapter_step, gen_s, n_tok, rank; written atomically. Trainer consumes FIFO (oldest adapter first)
      or --drop-stale (newest, older blocks discarded); consumed files are deleted.
  STOP                   trainer done -> rollout ranks exit.

Sampling rng: rollout rank r draws from numpy default_rng(seed*7919 + 1000 + r) -- per-rank streams,
NOT rl.py's every-rank-draws-the-same-B-then-slices scheme (there is no shared step any more). The first
--n-eval-dirs unique bank blocks are reserved exactly as in rl.py.

Metrics keep rl.py's names (reward/mean, policy/entropy, policy/kl_to_init, ratio/mean, ratio/clipfrac,
grad_norm, rollout/len_mean, time/step_s, var/*, rollouts/samples ...) plus
policy/offpolicy_lag_steps, policy/sampler_abs_dlogp, time/wait_rollouts_s, time/grad_sync_s,
rollout/queue_depth, rollout/gen_s, rollout/tok_per_s_per_replica, rollout/blocks_dropped.

ScaleRL variant (Khatri et al. 2025, "The Art of Scaling Reinforcement Learning Compute for LLMs", arXiv 2510.13786):
OPT-IN -- every flag below defaults to the legacy behaviour above (default run = bit-identical to before); --recipe scalerl
sets the whole bundle, flags given explicitly still win. Pure pieces are unit-tested on CPU in rl/test_rl_disagg_scalerl.py.
  --max-lag 8           PipelineRL-8: the rollout queue may hold 8 steps' worth of blocks, so a block can be up to ~8 policy
                        updates stale (legacy: 1). old_lp stays the SAMPLER's logprobs of the adapter that generated the
                        block, so the per-token IS weight below is the off-policy correction. Adapters swap between blocks
                        (a block = one <=96-token generate() call), not inside one, which is our grain of "in-flight" updates.
  --loss cispo          -sg(min(rho, eps_max)) * A * log pi   (truncated-IS REINFORCE, MiniMax-M1 / ScaleRL): no PPO clip of
                        the objective, every token keeps a gradient. --cispo-eps-max (paper ablates {4,5,8}: no difference).
  --loss-agg prompt     token-mean within each direction's G rollouts, then mean over directions (each prompt weighs 1).
  --adv-mode batch      (r - group_mean) / std of ALL surviving advantages in the global batch (rl.py's ScaleRL mode).
  --zero-var-filter     groups with identical rewards (std <= --zero-var-eps) leave the effective batch: loss weight 0 AND
                        out of every denominator (the paper's "effective batch"). Near-inert for a continuous cosine reward.
  --npr-threshold 0.9   No-Positive-Resampling, ADAPTED to the continuous reward: a rollout is a "positive" when its raw
                        cosine >= --npr-pass-cos; a direction whose cumulative pass rate over all its visits >= threshold is
                        dropped from all future sampling (trainer rank 0 publishes npr/dropped.json, rollout ranks exclude it).
  --autocast-bf16       trainer: policy forward under torch.autocast(bf16) with PEFT's LoRA input casting off -> bf16 LoRA
                        matmuls + bf16 saved activations (fp32 masters/AdamW, fp32 vocab math, inject hook, fp32 head unchanged)
  --fp32-head           trainer recomputes the frozen lm_head projection in fp32 (the log_softmax over the logits already
                        was fp32 -- see _chunked_logp; the vLLM sampler stays bf16-head / fp32-softmax, we cannot change it).
  --length-control      penalty (default, ALSO under --recipe scalerl: keeps the hinge LP) | interrupt (no LP; cap-hit
                        snippets are scored as generated = our analogue of the forced wrap-up; needs --trunc-reward unset).
  metrics under scalerl/*: is_weight_mean, is_trunc_frac, zero_var_dropped_frac, effective_groups, npr_* , lag_max,
  trunc_frac, step_skipped. Every pre-existing metric name is unchanged.

Launch inside the container (see modal_rl_disagg.py):
    python RL/rl_disagg.py --role launch --n-rollout 1 --n-trainer 3 --data-dir <pool> --init-adapter <sft> ...
"""
import argparse
import contextlib
import gc
import glob
import json
import math
import os
import re
import pickle
import shutil
import subprocess
import sys
import threading
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


# ----------------------------------------------------------------------------------------------
# args (superset of rl.py's flags so the v15 TRAIN_ARGS list can be reused verbatim)
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=("launch", "bench", "trainer", "rollout", "bench-rollout", "bench-trainer"), default="launch",
                    help="launch/bench = parent (spawns children); the rest are the per-GPU child roles")
    ap.add_argument("--n-rollout", type=int, default=1, help="X: vLLM rollout GPUs")
    ap.add_argument("--n-trainer", type=int, default=3, help="Y: HF trainer GPUs (DDP)")
    ap.add_argument("--work-dir", default="/tmp/disagg")
    ap.add_argument("--backend", default="nccl", choices=("gloo", "nccl"),
                    help="trainer DDP backend (nccl: GPU flat-buffer grad all-reduce; gloo = rl.py's CPU path, ~4.5 s for 2 GB)")
    ap.add_argument("--master-port", type=int, default=29611)
    # data / init / resume (rl.py)
    ap.add_argument("--data-dir", default="data/pretrain")
    ap.add_argument("--bank-file", default="vecs.f32")
    ap.add_argument("--direction-source", choices=("cluster", "random"), default="cluster")
    ap.add_argument("--n-eval-dirs", type=int, default=64)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--ref-adapter", default=None)
    ap.add_argument("--step-offset", type=int, default=0)
    ap.add_argument("--wandb-id", default=None)
    ap.add_argument("--save-dir", default="checkpoints/rl_disagg")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-steps", default="", help="comma-separated extra checkpoint steps, e.g. 25,40,60,90,130,200,300,450,675,1000 (log-spaced)")
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear LR warmup over the first N global steps (stability)")
    ap.add_argument("--run-name", default="mxf-rl-disagg")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # batch / sampling (rl.py)
    ap.add_argument("--groups-per-step", type=int, default=128, help="B: directions consumed per trainer step (global)")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--total-steps", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rollout-chunk", type=int, default=64, help="IGNORED (rl.py hf engine)")
    ap.add_argument("--logp-chunk", type=int, default=16, help="IGNORED (no HF old_logp recompute here)")
    ap.add_argument("--rollout-engine", default="vllm", help="IGNORED (always vllm)")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.85, help="rollout ranks own the GPU: 0.85-0.9")
    ap.add_argument("--vllm-logp-tol", type=float, default=0.10, help="IGNORED (see policy/sampler_abs_dlogp at step 0)")
    ap.add_argument("--micro-batch", type=int, default=0, help="0 = auto: largest of --mb-candidates that fits at max length")
    ap.add_argument("--mb-candidates", default="64,48,40,32,24,16,12,8,6,4")
    ap.add_argument("--ref-micro-batch", type=int, default=32)
    ap.add_argument("--score-batch", type=int, default=128)
    ap.add_argument("--vocab-chunk", type=int, default=32, help="positions per fp32 log_softmax chunk over the 248k vocab")
    # reward (rl.py)
    ap.add_argument("--reward-metric", choices=("proj", "cosine"), default="proj")
    # ---- modulation-lens objective (opt-in; leave --ar-reward blank for the maemm objective) ----
    ap.add_argument("--ar-reward", default="",
                    help="dir of the frozen text->modulation-vector AR. Set this to swap the "
                         "reward: the policy writes --bullets '*' lines for a real L42 activation, "
                         "each is mapped to its modulation vector by the AR, they are combined by "
                         "exact NON-NEGATIVE least squares, and the reward is the cosine of that "
                         "composition with the activation. Validated at 94% of the "
                         "measured-vector reference; retention is ~95% at every arity 1..12.")
    ap.add_argument("--ar-jlens", default="", help="lens.pt holding J[layer] (required with --ar-reward)")
    # Whitening the comparison space removes the target-blind component of the reward: a
    # constant direction is worth cos 0.343 unwhitened and 0.0064 whitened (measured, 20k rows),
    # and the first run duly spent one of four bullets on a fixed phrase. Applied to BOTH sides.
    ap.add_argument("--ar-whiten", default="", help="npz holding a J-space whitener (C^-1/2)")
    ap.add_argument("--ar-whiten-key", default="W_ridge0.1",
                    help="key inside --ar-whiten; must be a RIDGED inverse (eigvals.min()==0)")
    # Contrastive alternative to whitening: credit only the part of the fit that depends on
    # WHICH activation was injected. r_i = fit(b_i, t_i) - w * mean_j fit(b_i, t_{i+j+1}).
    ap.add_argument("--reward-contrast-negatives", type=int, default=0)
    ap.add_argument("--reward-contrast-weight", type=float, default=1.0)
    ap.add_argument("--ar-affine", default="",
                    help="the fitted atom->activation alignment (required with --ar-reward). "
                         "Atoms are modulation reads, targets are natural activations: two "
                         "different L42 distributions. Worth 1.76x FVE (0.360 -> 0.633).")
    ap.add_argument("--ar-amu", default="",
                    help="npz with 'mu', the activation-pool mean subtracted in J-space. TWO "
                         "different means is the measured configuration (one shared mean puts a "
                         "blank string at 0.259 cosine, two put it at 0.008).")
    ap.add_argument("--bullets", type=int, default=4,
                    help="'*' lines the policy writes. Marginal cosine per bullet collapses "
                         "(+0.085, +0.069, +0.053, +0.020 at k=2,4,8,12), so 4 is the corner.")
    ap.add_argument("--bullet-max-tok", type=int, default=12,
                    help="tokens per bullet; matches the dictionary's <=12-token cap")
    ap.add_argument("--inj-char", default="\u321c",
                    help="the injection marker character in --prompt-file. Default U+321C ('\u321c', "
                         "single token, id 158983), which is inv_core.INJ_CHAR -- the marker the AV "
                         "was SFT'd with, sitting inside <concept>...</concept>. Read it from the "
                         "source rather than guessing: U+3237 looks similar and yields zero hits.")
    ap.add_argument("--prompt-file", default="",
                    help="use THIS prompt instead of mxf.prompts.build_prompt_ids. Required when "
                         "warm-starting an adapter that was SFT'd on a different prompt -- a lens "
                         "must be read with the prompt it was trained on. The file holds the raw "
                         "job text; the chat template is applied here (add_generation_prompt=True, "
                         "enable_thinking=False) exactly as inv_train did, and the marker position "
                         "is located by scanning for the injection character. NOTE maemm puts its "
                         "marker LAST, arguing a mid-prompt marker leaves ~61 instruction tokens "
                         "after it and 'empirically erased conditioning'; our SFT prompt has the "
                         "marker at 40 of 186, so expect weaker conditioning than maemm's layout.")
    ap.add_argument("--reward-scale", type=float, default=1.0)
    ap.add_argument("--log-reward", action="store_true")
    ap.add_argument("--reward-window-last", type=int, default=0)
    ap.add_argument("--reward-topk", type=int, default=1)
    ap.add_argument("--reward-pos-penalty", type=float, default=0.0)
    ap.add_argument("--fluency-floor", type=float, default=-4.5)
    # Monitoring, not gating: sample the unconditioned-logp distribution so a floor can be set
    # from percentiles instead of a guess. A guessed -4.0 rejected 99.4% of legible rollouts.
    ap.add_argument("--flu-monitor-every", type=int, default=0,
                    help="log fluency/distinct percentiles every N steps on a subsample (0=off). "
                         "Costs one clean-base forward over --flu-monitor-n rollouts.")
    ap.add_argument("--flu-monitor-n", type=int, default=256)
    ap.add_argument("--no-fluency-floor", action="store_true",
                    help="disable the fluency gate entirely (keeps --distinct-floor, which then "
                         "costs no forward pass). Use with --flu-monitor-every to calibrate first.")
    ap.add_argument("--distinct-floor", type=float, default=0.5)
    ap.add_argument("--gate-penalty", type=float, default=25.0)
    ap.add_argument("--len-penalty-start", type=int, default=64)
    ap.add_argument("--len-penalty-per-tok", type=float, default=0.5)
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--no-len-penalty", action="store_true")
    # optimization (rl.py)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--tis-cap", type=float, default=2.0)
    ap.add_argument("--adv-mode", choices=["none", "group", "batch"], default=None)
    ap.add_argument("--loss-agg", choices=["token", "seq", "prompt"], default=None,
                    help="token (default) | seq | prompt (ScaleRL: token-mean within each direction's group, then mean over directions)")
    ap.add_argument("--trunc-reward", type=float, default=None)
    ap.add_argument("--adam-betas", type=float, nargs=2, default=(0.9, 0.999), metavar=("B1", "B2"))
    ap.add_argument("--grad-ckpt", action="store_true", help="IGNORED (never needed off-vLLM)")
    ap.add_argument("--std-norm", action="store_true")
    ap.add_argument("--batch-norm", action="store_true")
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--entropy-target", type=float, default=0.0,
                    help="adaptive entropy bonus: after each step coef *= exp(rate*(target - H)) so the policy's per-token "
                         "entropy is held near TARGET nats (0 = fixed --entropy-coef). Replaces the KL leash as the anti-collapse term.")
    ap.add_argument("--entropy-adapt-rate", type=float, default=0.05)
    ap.add_argument("--entropy-coef-max", type=float, default=0.05)
    ap.add_argument("--entropy-coef-min", type=float, default=1e-4)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--kl-cap", type=float, default=10.0)
    # inline eval flags accepted for launcher compatibility; see inline_eval_stub()
    ap.add_argument("--inline-eval-every", type=int, default=0)
    ap.add_argument("--eval-cache", default="/data/eval_universal_ho/eval_sets_heldout.pt")
    ap.add_argument("--eval-sae", default="/data/sae/ae.pt")
    ap.add_argument("--eval-bo", type=int, default=4)
    ap.add_argument("--eval-temp", type=float, default=1.0)
    ap.add_argument("--eval-max-new", type=int, default=64)
    ap.add_argument("--eval-min-new", type=int, default=16)
    ap.add_argument("--no-extra-evals", action="store_true")
    ap.add_argument("--eval-n-per-family", type=int, default=0)
    ap.add_argument("--transcript-every", type=int, default=5)
    ap.add_argument("--transcript-groups", type=int, default=4)
    ap.add_argument("--transcript-samples", type=int, default=4)
    ap.add_argument("--div-coef", type=float, default=0.0)
    ap.add_argument("--firsttok-coef", type=float, default=0.0)
    # disaggregation
    ap.add_argument("--rollout-block-groups", type=int, default=0,
                    help="directions per rollout block (per generate call); 0 = groups_per_step / n_rollout")
    ap.add_argument("--max-num-seqs", type=int, default=0, help="vLLM max_num_seqs; 0 = block_groups*group_size")
    ap.add_argument("--gdn-prefill-backend", choices=("triton", "flashinfer", "auto"), default="triton",
                    help="vLLM GatedDeltaNet PREFILL kernel. vLLM 0.19's 'auto' picks flashinfer's gdn_prefill on sm90 (H100/H200) -- a "
                         "JIT-compiled CUDA extension that needs nvcc/CUDA_HOME, absent from our image (EngineDeadError on 4xH200) -- and the "
                         "vendored fla Triton kernel everywhere else (what every B200 measurement ran). 'triton' = the Triton kernel on all "
                         "archs (default); 'auto' = vLLM's choice; 'flashinfer' = force the JIT kernel (needs nvcc in the image).")
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="vLLM FULL_DECODE_ONLY cudagraphs (compilation mode NONE) instead of the plugin-forced eager")
    ap.add_argument("--stock-lens-hook", action="store_true", help="use vllm_lens' stock O(reqs x keys x layers) hook (for A/B timing)")
    ap.add_argument("--max-queue-blocks", type=int, default=0,
                    help="rollout backpressure: max unconsumed blocks; 0 = one step's worth (lag stays 1 with full overlap)")
    ap.add_argument("--drop-stale", action="store_true", help="consume the NEWEST blocks and discard older ones (min lag, wastes rollouts)")
    ap.add_argument("--publish-every", type=int, default=1)
    ap.add_argument("--keep-loras", type=int, default=10,
                    help="published adapters kept: an eval request needs its adapter on disk until every rollout rank has generated its shard "
                         "(vLLM has 2 LoRA slots; the live policy advancing each step can evict the eval adapter -> reload from disk)")
    ap.add_argument("--publish-fp32", action="store_true",
                    help="publish the adapter in fp32 (rl.py behaviour). Default bf16: vLLM casts LoRA weights to the model dtype (bf16) on load anyway, so the served policy is identical and the write is half the size")
    ap.add_argument("--no-fla", action="store_true", help="trainer: block the fla (flash-linear-attention) GDN kernels -> HF torch fallback")
    # inline eval (disaggregated): rollout ranks generate the held-out eval texts as a side job, trainer ranks score
    ap.add_argument("--eval-chunk-seqs", type=int, default=512,
                    help="rollout ranks: eval sequences per generate() call when filling idle time (one chunk per idle slot)")
    ap.add_argument("--eval-max-delay-s", type=float, default=120.0,
                    help="rollout ranks: after this many seconds a pending eval request is worked on even if the rollout queue is not full")
    ap.add_argument("--eval-drop-after-s", type=float, default=2400.0,
                    help="trainer: an eval request whose shards have not all arrived after this long is dropped with an error")
    ap.add_argument("--bench-sizes", default="128,256,512,1024", help="bench-rollout: sequences per generate call")
    ap.add_argument("--bench-configs", default="eager:512,graphs:512,eager:1024,graphs:1024",
                    help="bench-rollout: <eager|graphs|stock>:<max_num_seqs> list, sharded over rollout ranks "
                         "(stock = eager with vllm_lens' stock hook, the rl.py baseline)")
    ap.add_argument("--bench-rollouts-per-rank", default="128,256,512,1024", help="bench-trainer: update() sizes to time")
    # ScaleRL variant (module docstring). default=None means "not given", so --recipe fills the bundle without clobbering
    # explicit flags; _resolve_recipe() turns the Nones into the legacy values otherwise.
    ap.add_argument("--recipe", choices=("", "scalerl"), default="", help="scalerl = the whole ScaleRL bundle (explicit flags win)")
    ap.add_argument("--loss", choices=("ppo", "cispo"), default=None,
                    help="ppo (default: rl.py's clipped surrogate on the TIS-capped ratio) | cispo (truncated-IS REINFORCE)")
    ap.add_argument("--cispo-eps-max", type=float, default=None, help="CISPO IS-weight truncation (paper ablates {4,5,8}; bundle 5)")
    ap.add_argument("--zero-var-filter", dest="zero_var_filter", action="store_true", default=None,
                    help="drop zero-variance groups from the effective batch (loss weight 0 and out of the denominators)")
    ap.add_argument("--no-zero-var-filter", dest="zero_var_filter", action="store_false")
    ap.add_argument("--zero-var-eps", type=float, default=1e-6, help="a group is zero-variance when its reward std <= this")
    ap.add_argument("--npr-threshold", type=float, default=None,
                    help="No-Positive-Resampling: drop a direction from future sampling once its cumulative pass rate >= this (0 = off)")
    ap.add_argument("--npr-pass-cos", type=float, default=0.7,
                    help="continuous-reward adaptation of 'correct': a rollout is a positive when its RAW cosine >= this")
    ap.add_argument("--max-lag", type=int, default=None,
                    help="max off-policyness in trainer steps: the rollout queue holds this many steps' worth of blocks "
                         "(an explicit --max-queue-blocks overrides); legacy 1, ScaleRL 8")
    ap.add_argument("--autocast-bf16", action="store_true",
                    help="trainer: run the POLICY forward under torch.autocast(bf16) with PEFT's LoRA input-dtype casting disabled, so the "
                         "LoRA matmuls and their saved activations are bf16 (the fp32 LoRA master weights + AdamW, the fp32 chunked "
                         "log-softmax, the inject hook and the fp32 head are unchanged). Off = byte-identical to the default path.")
    ap.add_argument("--fp32-head", dest="fp32_head", action="store_true", default=None,
                    help="trainer: recompute the frozen lm_head projection in fp32 (ScaleRL/MiniMax precision fix, trainer side)")
    ap.add_argument("--no-fp32-head", dest="fp32_head", action="store_false")
    ap.add_argument("--length-control", choices=("penalty", "interrupt"), default=None,
                    help="penalty = the --len-penalty-* hinge (default, also under --recipe scalerl) | interrupt = no length "
                         "penalty, cap-hit snippets scored as generated (our analogue of ScaleRL's forced interruption)")
    a = ap.parse_args(argv)
    assert a.div_coef == 0 and a.firsttok_coef == 0
    assert not (a.std_norm and a.batch_norm)
    assert a.temperature == 1.0, "T must be 1.0: the sampler's logprobs are the behaviour policy"
    if a.no_gates:
        a.fluency_floor = a.distinct_floor = None
    if a.no_len_penalty:
        a.len_penalty_start = None
    _resolve_recipe(a)
    if a.rollout_block_groups <= 0:
        a.rollout_block_groups = max(1, a.groups_per_step // max(a.n_rollout, 1))
    assert a.groups_per_step % a.rollout_block_groups == 0, "groups_per_step must be a multiple of rollout_block_groups"
    a.blocks_per_step = a.groups_per_step // a.rollout_block_groups
    if a.max_queue_blocks <= 0:
        a.max_queue_blocks = a.max_lag * a.blocks_per_step      # legacy max_lag 1 -> one step's worth (unchanged)
    if a.max_num_seqs <= 0:
        a.max_num_seqs = a.rollout_block_groups * a.group_size
    return a


# The two bundles _resolve_recipe() fills unspecified (None) variant flags from. LEGACY == the pre-ScaleRL defaults.
SCALERL_BUNDLE = {"loss": "cispo", "cispo_eps_max": 5.0, "loss_agg": "prompt", "zero_var_filter": True,
                  "npr_threshold": 0.9, "max_lag": 8, "fp32_head": True, "length_control": "penalty"}
LEGACY_BUNDLE = {"loss": "ppo", "cispo_eps_max": 5.0, "loss_agg": "token", "zero_var_filter": False,
                 "npr_threshold": 0.0, "max_lag": 1, "fp32_head": False, "length_control": "penalty"}


def _resolve_recipe(a):
    """Fill every ScaleRL-variant flag the user did not give (None) from the bundle --recipe selects; --recipe scalerl also
    picks --adv-mode batch unless an advantage mode was given. With --recipe '' and no variant flags nothing changes."""
    if getattr(a, "no_fluency_floor", False):
        a.fluency_floor = None
    bundle = SCALERL_BUNDLE if a.recipe == "scalerl" else LEGACY_BUNDLE
    for k, v in bundle.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.recipe == "scalerl" and a.adv_mode is None and not (a.std_norm or a.batch_norm):
        a.adv_mode = "batch"
    if a.length_control == "interrupt":
        a.len_penalty_start = None
        assert a.trunc_reward is None, "--length-control interrupt scores cap-hit rollouts as generated: unset --trunc-reward"
    assert a.cispo_eps_max > 0 and a.max_lag >= 1 and 0.0 <= a.npr_threshold <= 1.0
    assert a.max_lag == 1 or not a.drop_stale, "--max-lag > 1 needs the FIFO queue (--drop-stale would discard the lagged blocks)"
    return a


# ----------------------------------------------------------------------------------------------
# small shared helpers (filesystem protocol)
# ----------------------------------------------------------------------------------------------
def _atomic_write_text(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _read_latest(work):
    p = f"{work}/lora/latest"
    try:
        with open(p) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _queue_files(work):
    return sorted(glob.glob(f"{work}/queue/blk_*.pt"))


def _stop_requested(work):
    return os.path.exists(f"{work}/STOP")


def _log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def _bank_open(a):
    """Direction bank memmap + rl.py's eval-row reservation. Returns (bank, n_vecs, eval_rows)."""
    import numpy as np
    from mxf.config import D_MODEL
    stats_p = f"{a.data_dir}/build_stats.json"
    n_vecs = (json.load(open(stats_p))["n_examples"] if os.path.exists(stats_p)
              else os.path.getsize(f"{a.data_dir}/{a.bank_file}") // (4 * D_MODEL))
    bank = np.memmap(f"{a.data_dir}/{a.bank_file}", dtype=np.float32, mode="r", shape=(n_vecs, D_MODEL))
    eval_rows = 0
    if a.n_eval_dirs > 0:
        blocks, i = 0, 0
        while i < n_vecs and blocks < a.n_eval_dirs:
            row = np.asarray(bank[i]); i += 1; blocks += 1
            while i < n_vecs and np.array_equal(np.asarray(bank[i]), row):
                i += 1
        eval_rows = i
    return bank, n_vecs, eval_rows


# ----------------------------------------------------------------------------------------------
# ScaleRL pieces: pure functions / plain state, unit-tested on CPU in rl/test_rl_disagg_scalerl.py
# ----------------------------------------------------------------------------------------------
def _sample_block_idx(rng, lo, hi, size, dropped=None):
    """`size` distinct bank rows in [lo, hi), sorted. Without a drop set this is EXACTLY the original rollout draw
    (same rng stream); with one (No-Positive-Resampling) the dropped rows are excluded by rejection -- the surviving
    prefix of a uniform without-replacement draw is a uniform without-replacement draw from the allowed rows."""
    import numpy as np
    if not dropped:
        return lo + np.sort(rng.choice(hi - lo, size=size, replace=False))
    n_drop = sum(1 for i in dropped if lo <= i < hi)
    assert hi - lo - n_drop >= size, f"only {hi - lo - n_drop} directions left after NPR dropped {n_drop}"
    k = size
    while True:
        k = min(hi - lo, 2 * k)
        cand = lo + rng.choice(hi - lo, size=k, replace=False)
        cand = cand[np.fromiter((int(c) not in dropped for c in cand), dtype=bool, count=len(cand))]
        if len(cand) >= size:
            return np.sort(cand[:size])


class NPRTracker:
    """No-Positive-Resampling (ScaleRL: "maintaining a history of pass rates and permanently removing any prompt with pass
    rate >= 0.9 from subsequent epochs"), ADAPTED to a continuous reward: a rollout is a 'positive' when its raw cosine
    >= pass_cos; a direction's pass rate is positives / rollouts over ALL its visits so far (the "history"); once that rate
    >= threshold the direction is dropped from every future block (G=8, 0.9: 8/8 on one visit, 15/16 over two, ...).
    Owned by trainer rank 0 (which sees every rank's rewards); publish() writes the drop list _NPRDropList reads."""

    def __init__(self, threshold, pass_cos):
        self.threshold, self.pass_cos = float(threshold), float(pass_cos)
        self.hist = {}            # dir_idx -> [positives, rollouts]
        self.dropped = set()
        self._dirty = False

    def update(self, dir_idx, raw_cos, group_size):
        """dir_idx [B] bank rows (-1 = random direction, ignored); raw_cos [B*G] group-major RAW cosines. -> step stats."""
        import numpy as np
        idx = np.asarray(dir_idx).reshape(-1)
        pos = np.asarray(raw_cos, dtype=np.float64).reshape(len(idx), int(group_size)) >= self.pass_cos
        n_new, n_flag = 0, 0
        for i, row in zip(idx.tolist(), pos):
            if i < 0:
                continue
            h = self.hist.setdefault(i, [0, 0])
            h[0] += int(row.sum()); h[1] += int(row.size)
            if h[0] / h[1] >= self.threshold:
                n_flag += 1
                if i not in self.dropped:
                    self.dropped.add(i); n_new += 1; self._dirty = True
        return {"scalerl/npr_pass_frac": float(pos.mean()) if pos.size else 0.0,
                "scalerl/npr_batch_flagged_frac": n_flag / max(len(idx), 1),
                "scalerl/npr_new_dropped": float(n_new), "scalerl/npr_dropped_total": float(len(self.dropped)),
                "scalerl/npr_directions_seen": float(len(self.hist))}

    def publish(self, path):
        """Atomic json list of the dropped rows, rewritten only when the set changed. Returns True when written."""
        if not self._dirty:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_write_text(path, json.dumps(sorted(self.dropped)))
        self._dirty = False
        return True


class _NPRDropList:
    """Rollout-side reader of NPRTracker.publish(): re-reads the file only when its mtime changes."""

    def __init__(self, path):
        self.path, self.mtime, self.dropped = path, None, set()

    def refresh(self):
        try:
            mt = os.path.getmtime(self.path)
        except FileNotFoundError:
            return self.dropped
        if mt != self.mtime:
            try:
                with open(self.path) as f:
                    self.dropped = set(json.load(f))
                self.mtime = mt
            except (ValueError, OSError):     # mid-replace / half-written: keep the previous list, retry next block
                pass
        return self.dropped


def compute_advantages_disagg(r, n_groups, group_size, mode, zero_var_eps=1e-6, zero_var_filter=False):
    """rl.py compute_advantages (none = Dr. GRPO centering; group = / per-group std; batch = ScaleRL: zero-variance groups
    zeroed, / ONE std of all surviving advantages of the GLOBAL batch, all_reduce'd over DDP) with the zero-variance
    threshold exposed, plus the ScaleRL effective-batch mask: keep [n_groups*group_size] bool, None when the filter is
    off (-> the loss weights are the original ones bit for bit). Zero-variance = reward std <= zero_var_eps."""
    import torch
    import torch.distributed as dist
    rg = r.view(n_groups, group_size)
    adv = rg - rg.mean(1, keepdim=True)
    nz_g = rg.std(1) > zero_var_eps
    if mode == "group":
        adv = adv / (rg.std(1, keepdim=True) + 1e-6)
    elif mode == "batch":
        nz = nz_g[:, None].expand(-1, group_size)
        adv = adv * nz
        stats = torch.tensor([adv[nz].double().pow(2).sum(), adv[nz].double().sum(), nz.sum()], dtype=torch.float64)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(stats)                                        # CPU tensor -> gloo
        n = stats[2].item()
        std = math.sqrt(max(stats[0].item() / n - (stats[1].item() / n) ** 2, 0.0)) if n > 1 else 1.0
        adv = adv / (std + 1e-6)
    elif mode != "none":
        raise ValueError(mode)
    keep = nz_g.repeat_interleave(group_size) if zero_var_filter else None
    return adv.flatten().detach(), keep


def loss_weights(gen_mask, loss_agg, group_size=1, keep=None):
    """Per-token loss weights w_all [n,T] (sum 1 over the local batch) + the weight _sync_grads uses so the DDP
    all-reduce equals the single-GPU gradient over the union batch:
      token  : every completion token weighs 1/total_tok                              (sync_w = total_tok)  rl.py / DAPO
      seq    : every rollout weighs 1/n, its tokens 1/|y_i| within                    (sync_w = n)          GRPO sample-mean
      prompt : every GROUP (direction) weighs 1/n_groups, its tokens 1/sum_g |y_g| within (sync_w = n_groups) ScaleRL
    keep [n] bool (zero-variance filter): dropped rollouts weigh 0 and leave every denominator (effective batch).
    keep=None reproduces the original rl_disagg expressions bit for bit."""
    import torch
    n = gen_mask.shape[0]
    gm = gen_mask.float()
    if keep is None:
        if loss_agg == "seq":
            return gm / gen_mask.sum(1, keepdim=True).clamp(min=1).float() / n, float(n)
        if loss_agg == "token":
            total_tok = max(int(gen_mask.sum()), 1)
            return gm / total_tok, float(total_tok)
        keep = torch.ones(n, dtype=torch.bool)
    gm = gm * keep.float()[:, None]
    if loss_agg == "token":
        tot = max(int(gm.sum()), 1)
        return gm / tot, float(tot)
    if loss_agg == "seq":
        n_eff = max(int(keep.sum()), 1)
        return gm / gm.sum(1, keepdim=True).clamp(min=1) / n_eff, float(n_eff)
    if loss_agg == "prompt":
        G = int(group_size)
        assert n % G == 0, f"{n} rollouts is not a multiple of the group size {G}"
        m3 = gm.view(n // G, G, -1)
        tok_g = m3.sum((1, 2))                                            # completion tokens per group (0 when dropped)
        n_eff = max(int((tok_g > 0).sum()), 1)
        return (m3 / tok_g.clamp(min=1)[:, None, None] / n_eff).view(n, -1), float(n_eff)
    raise ValueError(loss_agg)


def pg_token_loss(new_lp, old_lp, A, loss, clip_eps, tis_cap, cispo_eps_max):
    """Per-token policy-gradient loss BEFORE the aggregation weights, for A broadcast over tokens ([n,1]):
      ppo   : rl.py's clipped surrogate on ratio = min(exp(new-old), tis_cap): -min(ratio*A, clip(ratio, 1-+eps)*A)
      cispo : ScaleRL / MiniMax-M1 truncated-IS REINFORCE: -sg(min(exp(new-old), eps_max)) * A * new_lp -- no clip of the
              objective, every token keeps a gradient; the truncation IS the off-policy correction (lower bound 0).
    Returns (loss_tok [n,T], ratio [n,T] = the IS weight the gradient sees (detached for cispo), rho [n,T] raw ratio, no grad)."""
    import torch
    if loss == "cispo":
        rho = torch.exp(new_lp.detach() - old_lp)
        is_w = rho.clamp(max=cispo_eps_max)
        return -(is_w * A * new_lp), is_w, rho
    ratio = torch.exp(new_lp - old_lp).clamp(max=tis_cap)
    with torch.no_grad():
        rho = torch.exp(new_lp - old_lp)
    return -torch.minimum(ratio * A, ratio.clamp(1 - clip_eps, 1 + clip_eps) * A), ratio, rho


def install_fp32_head(actor):
    """ScaleRL / MiniMax-M1 "FP32 at the LM head", trainer side: a forward hook replaces the lm_head output with
    F.linear(x.float(), W_fp32) so the logits the loss sees are not bf16-rounded (their log_softmax already was fp32, see
    _chunked_logp). W_fp32 is ONE persistent detached copy (2x the head's bf16 bytes; ~5 GB for a 248k x 5120 head) so no
    per-call cast is kept alive for backward. lm_head must be a frozen nn.Linear (PEFT 'all-linear' never wraps it).
    Returns the hook handle."""
    import torch
    import torch.nn.functional as F
    heads = [(n, m) for n, m in actor.named_modules() if n.endswith("lm_head")]
    assert len(heads) == 1, f"expected exactly one lm_head module, found {[n for n, _ in heads]}"
    name, head = heads[0]
    assert isinstance(head, torch.nn.Linear) and not any(p.requires_grad for p in head.parameters()), \
        f"{name} must be a frozen nn.Linear (got {type(head).__name__})"
    w32 = head.weight.detach().float()
    b32 = None if head.bias is None else head.bias.detach().float()

    def _fp32_out(mod, inp, out):
        with torch.autocast(inp[0].device.type, enabled=False):   # composable with --autocast-bf16: the head stays fp32 (no-op otherwise)
            return F.linear(inp[0].float(), w32, b32)
    return head.register_forward_hook(_fp32_out)


@contextlib.contextmanager
def _policy_precision(actor, enabled):
    """--autocast-bf16 region for the POLICY forward. Off: a pure no-op (default path byte-identical). On: (1) PEFT's LoRA
    input-dtype casting is disabled (peft.helpers.disable_input_dtype_casting, else the same `cast_input_dtype_enabled`
    toggle by hand) -- otherwise every LoRA layer still materialises an fp32 copy of its input; (2) torch.autocast(bf16) on the
    actor's device, so F.linear(x_bf16, W_lora_fp32) runs as a bf16 matmul and autograd saves bf16 activations. The LoRA
    master weights stay fp32 (grads arrive in fp32 through autocast's cast nodes), AdamW is untouched, rsLoRA `scaling` is a
    Python float applied after the matmul, lora_dropout is nn.Identity at p=0. The bf16 base layers are unaffected (their
    inputs are bf16 already; RMSNorm's explicit fp32 upcasts are explicit casts, which autocast does not override)."""
    if not enabled:
        yield
        return
    import torch
    dev = next(actor.parameters()).device.type
    try:
        from peft.helpers import disable_input_dtype_casting
        cm = disable_input_dtype_casting(actor)
    except ImportError:
        cm = _disable_input_dtype_casting_manual(actor)
    with cm, torch.autocast(dev, dtype=torch.bfloat16):
        yield


@contextlib.contextmanager
def _disable_input_dtype_casting_manual(model):
    """Fallback == peft.helpers.disable_input_dtype_casting: flip `cast_input_dtype_enabled` off on every tuner layer, restore after."""
    saved = {}
    for name, m in model.named_modules():
        if hasattr(m, "cast_input_dtype_enabled"):
            saved[name] = m.cast_input_dtype_enabled
            m.cast_input_dtype_enabled = False
    try:
        yield
    finally:
        for name, m in model.named_modules():
            if name in saved:
                m.cast_input_dtype_enabled = saved[name]


def _hook_outside_autocast(hook, enabled):
    """Wrap a forward hook so it runs with autocast DISABLED (the inject hook's ||h|| and add stay bf16-exact as in rl.py)."""
    if not enabled:
        return hook
    import torch

    def wrapped(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        with torch.autocast(h.device.type, enabled=False):
            return hook(mod, inp, out)
    return wrapped


# ----------------------------------------------------------------------------------------------
# inline-eval plumbing (pure functions; unit-tested on CPU in train/test_rl_disagg_queue.py)
#   eval_req/req_<k>.pt   trainer rank 0 -> everything the rollout side needs to generate ckpt k's eval texts
#   eval_gen/<k>_r<r>.pt  rollout rank r -> its row shard (rows i with i % X == r) of every eval set
# ----------------------------------------------------------------------------------------------
def _eval_req_path(work, k):
    return f"{work}/eval_req/req_{k:07d}.pt"


def _eval_shard_path(work, k, rank):
    return f"{work}/eval_gen/{k:07d}_r{rank}.pt"


def _eval_requests(work):
    """ckpt steps with a pending request, oldest first."""
    ks = []
    for f in glob.glob(f"{work}/eval_req/req_*.pt"):
        try:
            ks.append(int(os.path.basename(f)[4:-3]))
        except ValueError:
            pass
    return sorted(ks)


def _eval_shards_ready(work, k, n_rollout):
    return all(os.path.exists(_eval_shard_path(work, k, r)) for r in range(n_rollout))


def _eval_plan(req, rank, n_rollout, chunk_seqs):
    """Split ckpt k's eval generation into this rank's chunks (rows i % n_rollout == rank), each <= chunk_seqs
    sequences. Returns a list of dicts {set, kind, rows, n, temp, min_new, max_new, seeds} in a fixed order.
    Row -> seed: held-out families use eval_universal GEN_SEED*1000+i (== rl.py inline_eval), the extra-eval
    testbed uses snippet_locality GEN_SEED*1000+i mod 2^31-1 (== inline_extra_evals.run_extra_evals_gpu)."""
    chunks = []
    for st in req["sets"]:
        rows = [i for i in range(st["n_rows"]) if i % n_rollout == rank]
        per = max(1, chunk_seqs // max(int(st["n"]), 1))
        for c0 in range(0, len(rows), per):
            rr = rows[c0 : c0 + per]
            chunks.append({"set": st["name"], "kind": st["kind"], "rows": rr, "n": int(st["n"]), "temp": float(st["temp"]),
                           "min_new": int(st["min_new"]), "max_new": int(st["max_new"]),
                           "seeds": [int(st["seed_base"] + i) % 2147483647 for i in rr]})
    return chunks


def _eval_merge_shards(shards):
    """[{set: {row: [texts]}}, ...] -> {set: {row: [texts]}} (rows from all rollout ranks)."""
    out = {}
    for sh in shards:
        for name, rows in sh["texts"].items():
            out.setdefault(name, {}).update({int(i): v for i, v in rows.items()})
    return out


def _eval_sets_from_assets(EV, EX, a):
    """The eval 'sets' list for a request: one entry per held-out cosine family + the SAE family (+ the
    extra-eval testbed features). dirs are unit fp32 [n_rows, d]."""
    import torch.nn.functional as F
    EU, es = EV["EU"], EV["es"]
    sets = []
    for fam in EV["fams"]:
        du = F.normalize(es[f"{fam}_dirs"].float(), dim=-1)
        sets.append({"name": fam, "kind": "cos", "n_rows": int(du.shape[0]), "dirs": du, "n": a.eval_bo, "temp": a.eval_temp,
                     "min_new": a.eval_min_new, "max_new": a.eval_max_new, "seed_base": EU.GEN_SEED * 1000})
    du = F.normalize(es["sae_dirs"].float(), dim=-1)
    sets.append({"name": "sae", "kind": "sae", "n_rows": int(du.shape[0]), "dirs": du, "n": a.eval_bo, "temp": a.eval_temp,
                 "min_new": a.eval_min_new, "max_new": a.eval_max_new, "seed_base": EU.GEN_SEED * 1000, "feats": list(EV["feats"])})
    if EX is not None:
        import snippet_locality as SL
        cfg = EX["tb_config"]
        max_new = min(int(cfg.get("max_new", 64)), int(a.max_new_tokens))
        sets.append({"name": "extra", "kind": "extra", "n_rows": len(EX["feats"]), "dirs": F.normalize(EX["dirs"].float(), dim=-1),
                     "n": int(EX["n_rollouts"]), "temp": float(cfg.get("temp", 1.0)), "min_new": min(int(cfg.get("min_new", 16)), max_new),
                     "max_new": max_new, "seed_base": SL.GEN_SEED * 1000, "feats": list(EX["feats"])})
    return sets


class _GenCfgStub:
    """rl.py's _eos_ids(tok, actor) only reads actor.generation_config -- rollout ranks have no actor."""
    def __init__(self, gen_cfg):
        self.generation_config = gen_cfg


# ==============================================================================================
# ROLLOUT rank
# ==============================================================================================
def _build_engine(a, rank, p_len, max_seqs, use_graphs, tag):
    """vLLM engine on this rank's (only visible) GPU. vllm_lens' plugin is loaded first so its
    LLM.generate/steering patches are live; its EngineArgs patch (which FORCES enforce_eager and
    installs the slow stock hook) is replaced by ours: fast_lens_ext + our own eager/graph choice."""
    hidden = {k: os.environ.pop(k) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                                             "ROLE_RANK", "MASTER_ADDR", "MASTER_PORT") if k in os.environ}
    try:
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        import vllm_lens._activations_plugin as P
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
        orig = P._original_create_engine_config
        assert orig is not None, "vllm_lens plugin did not register (dist-info missing?)"
        ext_cls = None if a.stock_lens_hook else "fast_lens_ext.FastSteerExtension"

        def _cfg(self, *args, **kw):
            if ext_cls and not self.worker_extension_cls:
                self.worker_extension_cls = ext_cls
            if not self.worker_extension_cls:
                self.worker_extension_cls = "vllm_lens._worker_ext.HiddenStatesExtension"
            return orig(self, *args, **kw)
        EngineArgs.create_engine_config = _cfg
        from mxf.config import MODEL
        max_len = p_len + a.max_new_tokens + 8
        kw = dict(model=MODEL, tensor_parallel_size=1, gpu_memory_utilization=a.vllm_gpu_mem, max_model_len=max_len,
                  attention_backend="TRITON_ATTN", language_model_only=True, enable_prefix_caching=False,
                  enable_lora=True, max_loras=2, max_lora_rank=64, max_num_seqs=int(max_seqs),   # 2 slots: live policy + eval ckpt
                  # never chunk a prompt (the marker must be prefilled in a hooked, eager pass): budget = every seq's full prompt+gen
                  # unless overridden (the eval daemon uses a smaller budget so the profiling run leaves KV memory for concurrency)
                  max_num_batched_tokens=int(getattr(a, "max_num_batched_tokens", 0) or 0) or max(8192, int(max_seqs) * max_len),
                  seed=a.seed * 1000 + 500 + rank, dtype="bfloat16")
        gdn = str(getattr(a, "gdn_prefill_backend", "triton") or "triton").lower()
        if gdn != "auto":   # EngineArgs.gdn_prefill_backend -> additional_config["gdn_prefill_backend"] -> ChunkGatedDeltaRule (sm90: flashinfer JIT unless 'triton')
            kw["gdn_prefill_backend"] = gdn
        if use_graphs:
            kw["enforce_eager"] = False
            kw["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                        "max_cudagraph_capture_size": int(max_seqs)}
        else:
            kw["enforce_eager"] = True
        t0 = time.time()
        llm = LLM(**kw)
        llm.collective_rpc("install_hooks")
        _log(tag, f"engine up in {time.time() - t0:.0f}s | graphs={use_graphs} max_num_seqs={max_seqs} "
                  f"mem={a.vllm_gpu_mem} ext={'stock' if a.stock_lens_hook else 'fast'} gdn_prefill={gdn}")
        return llm
    finally:
        os.environ.update(hidden)


def _verify_injection(llm, prompt_ids, marker, hnorm, tag, seed=0):
    """verify_vllm_injection without an HF actor: the injected vector is ABSOLUTE (norm_match=False), so
    the captured marker-row delta must equal hnorm*STEER_COEFF*unit(v): cos>0.99, ratio in [0.95,1.05];
    pre-marker rows untouched. Runs on the BASE weights (no LoRA request), greedy, 1 token."""
    import torch
    import torch.nn.functional as F
    import rl_hf as R
    from vllm import SamplingParams
    from mxf.config import D_MODEL, INJECT_LAYER, STEER_COEFF
    g = torch.Generator().manual_seed(seed)
    v = F.normalize(torch.randn(D_MODEL, generator=g), dim=0)

    def run(steer):
        extra = {"output_residual_stream": [INJECT_LAYER]}
        if steer:
            extra["apply_steering_vectors"] = [R._steer_vec(v, hnorm, marker)]
        out = llm.generate([{"prompt_token_ids": list(prompt_ids)}],
                           [SamplingParams(temperature=0.0, max_tokens=1, extra_args=extra)], use_tqdm=False)[0]
        act = getattr(out, "activations", None)
        assert act is not None and "residual_stream" in act, "capture returned nothing -- hooks not live?"
        return act["residual_stream"][0].float()
    h_clean, h_steer = run(False), run(True)
    delta = h_steer[marker] - h_clean[marker]
    cos = F.cosine_similarity(delta, v, dim=0).item()
    ratio = (delta.norm() / (STEER_COEFF * hnorm)).item()
    other = (h_steer[:marker] - h_clean[:marker]).norm(dim=-1).max().item() if marker > 0 else 0.0
    chk = {"cos": cos, "norm_ratio": ratio, "hnorm_published": hnorm, "hnorm_vllm_base": h_clean[marker].norm().item(),
           "max_other_row_delta": other, "ok": cos > 0.99 and 0.95 < ratio < 1.05}
    _log(tag, f"injection check: cos={cos:.4f} ratio={ratio:.3f} ||h||_vllm_base={chk['hnorm_vllm_base']:.1f} "
              f"(published adapter-on {hnorm:.1f}) pre-marker max|d|={other:.2e} -> {'OK' if chk['ok'] else 'FAIL'}")
    return chk


def _generate_block(llm, a, tok, prompt_ids, marker, dirs, hnorm, lora_req, eos_ids, key_prefix):
    """ONE generate() call: one request per direction, n=G, steering keyed by _steering_id (one RPC for
    the whole block). Returns gen_ids (group-major, stop token kept via _trim_at_stop), per-token vLLM
    logprobs (None where the engine dropped the stop token and we re-appended it), appended count, gen_s."""
    import rl_hf as R
    from vllm import SamplingParams
    G = a.group_size
    keys = [f"{key_prefix}_{i}" for i in range(len(dirs))]
    payload = {k: [R._steer_vec(v, hnorm, marker)] for k, v in zip(keys, dirs)}
    if a.stock_lens_hook:   # stock plugin protocol: per-request apply_steering_vectors (it does the RPCs itself)
        params = [SamplingParams(n=G, temperature=a.temperature, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                                 max_tokens=a.max_new_tokens, min_tokens=a.min_new_tokens, stop_token_ids=sorted(eos_ids),
                                 logprobs=0, extra_args={"apply_steering_vectors": payload[k]}) for k in keys]
    else:
        llm.collective_rpc("set_steering_data_many", args=(pickle.dumps(payload),))
        params = [SamplingParams(n=G, temperature=a.temperature, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                                 max_tokens=a.max_new_tokens, min_tokens=a.min_new_tokens, stop_token_ids=sorted(eos_ids),
                                 logprobs=0, extra_args={"_steering_id": k}) for k in keys]
    reqs = [{"prompt_token_ids": list(prompt_ids)} for _ in keys]
    t1 = time.time()
    try:
        outs = llm.generate(reqs, params, lora_request=lora_req, use_tqdm=False)
    finally:
        if not a.stock_lens_hook:
            llm.collective_rpc("clear_steering_data_many", args=(keys,))
    gen_s = time.time() - t1
    gen_ids, lps, appended = [], [], 0
    for out in outs:
        assert len(out.outputs) == G, f"expected {G} samples, got {len(out.outputs)}"
        for o in out.outputs:
            g = list(o.token_ids)
            lp = [None] * len(g)
            if o.logprobs:
                lp = [(d[t].logprob if (d is not None and t in d) else None) for d, t in zip(o.logprobs, g)]
            if o.finish_reason == "stop" and (not g or g[-1] not in eos_ids):
                g.append(int(o.stop_reason) if isinstance(o.stop_reason, int) else int(tok.eos_token_id))
                lp.append(None); appended += 1
            g2 = R._trim_at_stop(g, eos_ids)
            gen_ids.append(g2); lps.append(lp[: len(g2)])
    return gen_ids, lps, appended, gen_s


def _generate_eval_chunk(llm, a, tok, prompt_ids, marker, ch, dirs, hnorm, lora_req, eos_ids, key_prefix):
    """One generate() for an eval chunk: n samples per row at the set's temperature / token limits with the
    per-row seeds (deterministic like rl.py inline_eval), steering via _steering_id (one RPC per chunk).
    Returns {row: [n texts]} (stop token trimmed, decoded, empty -> ' ' as in inline_extra_evals)."""
    import rl_hf as R
    from vllm import SamplingParams
    keys = [f"{key_prefix}_{i}" for i in ch["rows"]]
    payload = {k: [R._steer_vec(dirs[i], hnorm, marker)] for k, i in zip(keys, ch["rows"])}
    llm.collective_rpc("set_steering_data_many", args=(pickle.dumps(payload),))
    params = [SamplingParams(n=ch["n"], temperature=ch["temp"], top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                             max_tokens=ch["max_new"], min_tokens=ch["min_new"], stop_token_ids=sorted(eos_ids), seed=sd,
                             extra_args={"_steering_id": k}) for k, sd in zip(keys, ch["seeds"])]
    reqs = [{"prompt_token_ids": list(prompt_ids)} for _ in keys]
    try:
        outs = llm.generate(reqs, params, lora_request=lora_req, use_tqdm=False)
    finally:
        llm.collective_rpc("clear_steering_data_many", args=(keys,))
    res = {}
    for i, out in zip(ch["rows"], outs):
        assert len(out.outputs) == ch["n"], f"expected {ch['n']} samples, got {len(out.outputs)}"
        res[i] = [(tok.decode(R._trim_at_stop(list(o.token_ids), eos_ids), skip_special_tokens=True).strip() or " ")
                  for o in out.outputs]
    return res


class _EvalJob:
    """Rollout-side state of one eval request: the chunks still to generate + the texts generated so far."""

    def __init__(self, work, k, rank, n_rollout, a, tag):
        import torch
        from vllm.lora.request import LoRARequest
        self.k, self.rank, self.tag, self.work = k, rank, tag, work
        self.req = torch.load(_eval_req_path(work, k), weights_only=False)
        self.adapter_step = int(self.req["adapter_step"])
        d = f"{work}/lora/step_{self.adapter_step}"
        self.error = None if os.path.isdir(d) else f"adapter step {self.adapter_step} no longer published"
        self.hnorm = float(json.load(open(f"{d}/meta.json"))["hnorm"]) if self.error is None else None
        self.lora_req = LoRARequest(lora_name=f"step{self.adapter_step}", lora_int_id=self.adapter_step + 1, lora_path=d)
        self.dirs = {st["name"]: st["dirs"] for st in self.req["sets"]}
        self.chunks = _eval_plan(self.req, rank, n_rollout, a.eval_chunk_seqs)
        self.texts = {st["name"]: {} for st in self.req["sets"]}
        self.t_gen, self.t0, self.n_seq = 0.0, time.time(), 0
        _log(tag, f"eval request ckpt {k}: {len(self.chunks)} chunks / {sum(len(c['rows']) * c['n'] for c in self.chunks)} seqs for this rank"
                  + (f" | ERROR {self.error}" if self.error else ""))

    @property
    def age(self):
        return time.time() - float(self.req.get("t", self.t0))

    def done(self):
        return self.error is not None or not self.chunks

    def step(self, llm, a, tok, prompt_ids, marker, eos_ids):
        ch = self.chunks.pop(0)
        t1 = time.time()
        try:
            res = _generate_eval_chunk(llm, a, tok, prompt_ids, marker, ch, self.dirs[ch["set"]], self.hnorm, self.lora_req, eos_ids,
                                       key_prefix=f"ev{self.k}r{self.rank}{ch['set']}")
            self.texts[ch["set"]].update(res)
            self.n_seq += len(ch["rows"]) * ch["n"]
        except Exception as e:  # noqa
            self.error = f"rank{self.rank}: {type(e).__name__}: {str(e)[:300]}"
            self.chunks = []
        self.t_gen += time.time() - t1

    def write(self):
        import torch
        fn = _eval_shard_path(self.work, self.k, self.rank)
        torch.save({"ckpt_step": self.k, "adapter_step": self.adapter_step, "rank": self.rank, "texts": self.texts,
                    "t_gen": self.t_gen, "n_seq": self.n_seq, "error": self.error, "t_done": time.time()}, fn + ".tmp")
        os.replace(fn + ".tmp", fn)
        _log(self.tag, f"eval shard ckpt {self.k}: {self.n_seq} seqs in {self.t_gen:.1f}s gen, {self.age:.0f}s after the request"
                       + (f" | ERROR {self.error}" if self.error else ""))


def run_rollout(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, GenerationConfig
    import rl_hf as R
    from mxf.config import D_MODEL, MODEL
    from mxf.prompts import build_prompt_ids
    from vllm.lora.request import LoRARequest

    rank = int(os.environ["DISAGG_RANK"]); tag = f"R{rank}"
    work = a.work_dir
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.prompt_file:
        _job = open(a.prompt_file).read()
        _txt = tok.apply_chat_template([{"role": "user", "content": _job}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok.encode(_txt, add_special_tokens=False)
        _inj = tok(a.inj_char, add_special_tokens=False).input_ids
        if len(_inj) != 1:
            raise SystemExit("--inj-char %r must be a SINGLE token, got %d: %s"
                             % (a.inj_char, len(_inj), _inj))
        _hits = [i for i, t in enumerate(prompt_ids) if t == _inj[0]]
        if len(_hits) != 1:
            raise SystemExit("--prompt-file must contain EXACTLY one %r (token id %d), found %d. "
                             "The AV was SFT'd with inv_core.INJ_CHAR = U+321C; a different char "
                             "means the injection would land in the wrong place."
                             % (a.inj_char, _inj[0], len(_hits)))
        mpos = [_hits[0]]
        # neighbour check, as inv_train asserts: the marker sits inside <concept>...</concept>, so
        # a shifted position would inject into the tag rather than the slot.
        _lo = tok("<concept>", add_special_tokens=False).input_ids
        _hi = tok("</concept>", add_special_tokens=False).input_ids
        _k = mpos[0]
        if not (prompt_ids[_k - len(_lo):_k] == _lo and
                prompt_ids[_k + 1:_k + 1 + len(_hi)] == _hi):
            _log(tag, "WARNING marker neighbours are not <concept>/</concept>; injection may be "
                      "misplaced relative to SFT")
        _log(tag, "prompt from %s: %d tokens, marker at %d (%d tokens follow it) -- maemm's own "
                  "layout puts the marker last; a mid-prompt marker is reported to weaken "
                  "conditioning" % (a.prompt_file, len(prompt_ids), mpos[0],
                                    len(prompt_ids) - 1 - mpos[0]))
    else:
        prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    eos_ids = R._eos_ids(tok, _GenCfgStub(GenerationConfig.from_pretrained(MODEL)))
    rng = np.random.default_rng(a.seed * 7919 + 1000 + rank)
    bank = n_vecs = None; eval_rows = 0
    if a.direction_source == "cluster":
        bank, n_vecs, eval_rows = _bank_open(a)
        assert n_vecs - eval_rows >= a.rollout_block_groups
    Bb = a.rollout_block_groups

    # the engine can load while the trainer is still loading the actor; the first block waits for step 0
    llm = _build_engine(a, rank, p_len, a.max_num_seqs, a.cuda_graphs, tag)
    t_wait = time.time()
    while _read_latest(work) is None:
        if _stop_requested(work):
            return
        time.sleep(1.0)
    _log(tag, f"first adapter published after {time.time() - t_wait:.0f}s of waiting")
    cur_step, lora_req, hnorm = None, None, None

    def refresh():
        nonlocal cur_step, lora_req, hnorm
        k = _read_latest(work)
        if k is None or k == cur_step:
            return False
        d = f"{work}/lora/step_{k}"
        meta = json.load(open(f"{d}/meta.json"))
        hnorm = float(meta["hnorm"])
        lora_req = LoRARequest(lora_name=f"step{k}", lora_int_id=k + 1, lora_path=d)
        cur_step = k
        return True
    refresh()
    chk = _verify_injection(llm, prompt_ids, marker, hnorm, tag, seed=a.seed)
    json.dump(chk, open(f"{work}/verify_r{rank}.json", "w"))
    if not chk["ok"]:
        raise RuntimeError(f"vLLM steering does NOT match the HF inject hook: {chk}")

    blk = 0
    inflight = f"{work}/queue/.inflight_{rank}"
    n_rollout = int(os.environ["DISAGG_WORLD"])
    ev_job, ev_done = None, set()
    npr_drop = _NPRDropList(f"{work}/npr/dropped.json") if a.npr_threshold > 0 else None   # No-Positive-Resampling

    def depth():   # complete blocks + blocks other ranks are generating right now (so N producers cannot all overshoot the cap)
        return len(_queue_files(work)) + len([f for f in glob.glob(f"{work}/queue/.inflight_*") if f != inflight])

    def eval_job():
        """The oldest pending eval request this rank has not finished (None if there is none)."""
        nonlocal ev_job
        if ev_job is None:
            for k in _eval_requests(work):
                if k not in ev_done and not os.path.exists(_eval_shard_path(work, k, rank)):
                    try:
                        ev_job = _EvalJob(work, k, rank, n_rollout, a, tag)
                    except Exception as e:  # noqa — request file half-written / adapter vanished: retry next slot
                        _log(tag, f"eval request ckpt {k} not loadable yet ({type(e).__name__}: {e})")
                    break
        return ev_job
    while not _stop_requested(work):
        # ---- eval generation fills the slack: whenever the rollout queue is full (the trainer is the bottleneck) do ONE
        # chunk of pending eval work instead of sleeping; a request older than --eval-max-delay-s is worked on regardless ----
        job = eval_job()
        if job is not None and (depth() >= a.max_queue_blocks or job.age > a.eval_max_delay_s):
            if not job.done():
                job.step(llm, a, tok, prompt_ids, marker, eos_ids)
            if job.done():
                job.write(); ev_done.add(job.k); ev_job = None
            continue
        while depth() >= a.max_queue_blocks:                           # backpressure: bounded lag, no wasted rollouts
            if _stop_requested(work):
                return
            if eval_job() is not None:
                break                                                  # -> top of the loop: eval chunk instead of idling
            time.sleep(0.25 + 0.05 * rank)
        if depth() >= a.max_queue_blocks:
            continue
        open(inflight, "w").close()
        t0 = time.time()
        swapped = refresh()
        if a.direction_source == "random":
            idx = np.full(Bb, -1, dtype=np.int64)
            dirs = F.normalize(torch.randn(Bb, D_MODEL, dtype=torch.float32), dim=-1)
        else:
            idx = _sample_block_idx(rng, eval_rows, n_vecs, Bb, npr_drop.refresh() if npr_drop is not None else None)
            dirs = F.normalize(torch.from_numpy(np.asarray(bank[idx], dtype=np.float32)), dim=-1)
        gen_ids, lps, appended, gen_s = _generate_block(llm, a, tok, prompt_ids, marker, dirs, hnorm, lora_req, eos_ids,
                                                        key_prefix=f"r{rank}b{blk}")
        n_tok = sum(len(g) for g in gen_ids)
        rec = {"block": blk, "rank": rank, "adapter_step": cur_step, "dir_idx": idx, "dirs": dirs, "gen_ids": gen_ids,
               "lps": lps, "appended": appended, "gen_s": gen_s, "n_tok": n_tok, "t_done": time.time(),
               "lora_swapped": swapped}
        fn = f"{work}/queue/blk_{cur_step:07d}_{time.time_ns()}_{rank}.pt"
        torch.save(rec, fn + ".tmp"); os.replace(fn + ".tmp", fn)
        try:
            os.remove(inflight)
        except FileNotFoundError:
            pass
        _log(tag, f"block {blk} | adapter step {cur_step}{' (swapped)' if swapped else ''} | {len(gen_ids)} seqs "
                  f"{n_tok} tok in gen {gen_s:.1f}s ({n_tok / gen_s:.0f} tok/s, {len(gen_ids) / gen_s:.1f} seq/s) "
                  f"| appended_stop {appended} | wall {time.time() - t0:.1f}s | queue {len(_queue_files(work))}")
        blk += 1
    _log(tag, "STOP seen, exiting")


def run_bench_rollout(a):
    """Rollout throughput table: for each <eager|graphs>:<max_num_seqs> config assigned to this rank
    (configs[rank::n_rollout]) build an engine, verify injection, then time generate() for every size in
    --bench-sizes (fresh directions, the SFT adapter converted to vLLM layout, real sampling params).
    Writes <work>/bench_rollout_r<rank>.json."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, GenerationConfig
    import rl_hf as R
    from mxf.config import MODEL
    from mxf.prompts import build_prompt_ids
    from vllm.lora.request import LoRARequest

    rank = int(os.environ["DISAGG_RANK"]); world = int(os.environ["DISAGG_WORLD"]); tag = f"B{rank}"
    work = a.work_dir
    tok = AutoTokenizer.from_pretrained(MODEL)
    if a.prompt_file:
        _job = open(a.prompt_file).read()
        _txt = tok.apply_chat_template([{"role": "user", "content": _job}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok.encode(_txt, add_special_tokens=False)
        _inj = tok(a.inj_char, add_special_tokens=False).input_ids
        if len(_inj) != 1:
            raise SystemExit("--inj-char %r must be a SINGLE token, got %d: %s"
                             % (a.inj_char, len(_inj), _inj))
        _hits = [i for i, t in enumerate(prompt_ids) if t == _inj[0]]
        if len(_hits) != 1:
            raise SystemExit("--prompt-file must contain EXACTLY one %r (token id %d), found %d. "
                             "The AV was SFT'd with inv_core.INJ_CHAR = U+321C; a different char "
                             "means the injection would land in the wrong place."
                             % (a.inj_char, _inj[0], len(_hits)))
        mpos = [_hits[0]]
        # neighbour check, as inv_train asserts: the marker sits inside <concept>...</concept>, so
        # a shifted position would inject into the tag rather than the slot.
        _lo = tok("<concept>", add_special_tokens=False).input_ids
        _hi = tok("</concept>", add_special_tokens=False).input_ids
        _k = mpos[0]
        if not (prompt_ids[_k - len(_lo):_k] == _lo and
                prompt_ids[_k + 1:_k + 1 + len(_hi)] == _hi):
            _log(tag, "WARNING marker neighbours are not <concept>/</concept>; injection may be "
                      "misplaced relative to SFT")
        _log(tag, "prompt from %s: %d tokens, marker at %d (%d tokens follow it) -- maemm's own "
                  "layout puts the marker last; a mid-prompt marker is reported to weaken "
                  "conditioning" % (a.prompt_file, len(prompt_ids), mpos[0],
                                    len(prompt_ids) - 1 - mpos[0]))
    else:
        prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    eos_ids = R._eos_ids(tok, _GenCfgStub(GenerationConfig.from_pretrained(MODEL)))
    # SFT adapter -> vLLM key layout (no actor needed: pure key rename, cf. rl.py _save_adapter_for_vllm)
    lora_dir = f"{work}/bench_lora_r{rank}"
    os.makedirs(lora_dir, exist_ok=True)
    sd = load_file(f"{a.init_adapter}/adapter_model.safetensors")
    out = {}
    for k, v in sd.items():
        k2 = k if "language_model" in k else k.replace("model.layers.", "model.language_model.layers.", 1)
        out[k2] = v.contiguous()
    save_file(out, f"{lora_dir}/adapter_model.safetensors", metadata={"format": "pt"})
    shutil.copy(f"{a.init_adapter}/adapter_config.json", f"{lora_dir}/adapter_config.json")
    lora_req = LoRARequest(lora_name="bench", lora_int_id=1, lora_path=lora_dir)
    bank, n_vecs, eval_rows = _bank_open(a)
    rng = np.random.default_rng(a.seed + rank)
    configs = [c for c in a.bench_configs.split(",") if c][rank::max(world, 1)]
    sizes = [int(s) for s in a.bench_sizes.split(",")]
    results = []
    _log(tag, f"configs for this rank: {configs} | sizes {sizes}")
    for cfg in configs:
        mode, mns = cfg.split(":"); mns = int(mns)
        a.stock_lens_hook = (mode == "stock")
        llm = _build_engine(a, rank, p_len, mns, mode == "graphs", tag)
        # base-weights clean marker norm from a capture is a fine stand-in for the trainer's published hnorm here
        from vllm import SamplingParams
        from mxf.config import INJECT_LAYER
        o = llm.generate([{"prompt_token_ids": list(prompt_ids)}],
                         [SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [INJECT_LAYER]})],
                         use_tqdm=False)[0]
        hnorm = o.activations["residual_stream"][0].float()[marker].norm().item()
        chk = _verify_injection(llm, prompt_ids, marker, hnorm, tag, seed=a.seed)
        for n_seqs in sizes:
            if n_seqs > mns * 4:      # more than 4 waves of the engine's capacity is not a config we would run
                continue
            Bb = max(1, n_seqs // a.group_size)
            idx = eval_rows + np.sort(rng.choice(n_vecs - eval_rows, size=Bb, replace=False))
            dirs = F.normalize(torch.from_numpy(np.asarray(bank[idx], dtype=np.float32)), dim=-1)
            # warm-up (LoRA load + graph warm) then the timed call
            _generate_block(llm, a, tok, prompt_ids, marker, dirs[: max(1, Bb // 4)], hnorm, lora_req, eos_ids, f"warm{n_seqs}")
            gen_ids, lps, appended, gen_s = _generate_block(llm, a, tok, prompt_ids, marker, dirs, hnorm, lora_req, eos_ids, f"bench{n_seqs}")
            n_tok = sum(len(g) for g in gen_ids)
            stats = None
            try:
                stats = llm.collective_rpc("fast_lens_stats")[0] if not a.stock_lens_hook else None
            except Exception:  # noqa
                pass
            row = {"mode": mode, "max_num_seqs": mns, "n_seqs": len(gen_ids), "gen_s": gen_s, "n_tok": n_tok,
                   "tok_per_s": n_tok / gen_s, "seq_per_s": len(gen_ids) / gen_s, "len_mean": n_tok / len(gen_ids),
                   "appended_stop": appended, "verify": chk, "lens_stats": stats,
                   "sample": tok.decode(gen_ids[0], skip_special_tokens=True)[:100]}
            results.append(row)
            _log(tag, f"{mode} mns={mns} n={len(gen_ids)}: {gen_s:.1f}s -> {row['tok_per_s']:.0f} tok/s, {row['seq_per_s']:.1f} seq/s, "
                      f"len {row['len_mean']:.1f} | appended {appended} | lens {stats}")
            json.dump(results, open(f"{work}/bench_rollout_r{rank}.json", "w"), indent=1)
        del llm
        gc.collect(); torch.cuda.empty_cache()
        time.sleep(3)
    _log(tag, "bench done")


# ==============================================================================================
# TRAINER rank
# ==============================================================================================
def _sync_grads(params, weight, backend, device):
    """Exact weighted all-reduce of the LoRA grads (one flat buffer): global grad = sum_r w_r g_r / sum_r w_r.
    weight = local rollout count (seq-mean loss) or local completion tokens (token-mean loss) -> equals the
    single-GPU gradient over the union batch even with UNEVEN shards. GPU buffer under nccl, CPU under gloo."""
    import torch
    import torch.distributed as dist
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    dev = device if backend == "nccl" else "cpu"
    flat = torch.cat([g.detach().reshape(-1).float() for g in grads] + [torch.ones(1, device=grads[0].device)]).to(dev)
    flat.mul_(float(weight))
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    tot = flat[-1].item()
    if tot <= 0:                  # every rank's effective batch is empty (zero-variance filter): nothing to average
        for g in grads:
            g.zero_()
        return 0.0
    flat.div_(tot)
    off = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[off : off + n].view_as(g))
        off += n
    return tot


def _chunked_logp(logits, targets, vocab_chunk, need_entropy_grad):
    """log_softmax over the 248k vocab in fp32, `vocab_chunk` positions at a time (bounds the transient to
    mb*chunk*V*4 bytes). Returns new_lp [mb,T] (with grad) and per-token entropy [mb,T] (no grad unless asked).
    Math identical to rl.py: log_softmax(logits.float()).gather(targets)."""
    import torch
    T = logits.shape[1]
    lp_chunks, ent_chunks = [], []
    for c0 in range(0, T, vocab_chunk):
        c1 = min(c0 + vocab_chunk, T)
        lpf = torch.log_softmax(logits[:, c0:c1].float(), -1)
        lp_chunks.append(lpf.gather(-1, targets[:, c0:c1, None]).squeeze(-1))
        if need_entropy_grad:
            ent_chunks.append(-(lpf.exp() * lpf).sum(-1))
        else:
            with torch.no_grad():
                ent_chunks.append(-(lpf.exp() * lpf).sum(-1))
        del lpf
    return torch.cat(lp_chunks, 1), torch.cat(ent_chunks, 1)


def update_disagg(actor, opt, submodule, ids, attn, p_len, marker, old_lp, known, adv, dirs_rep, a, device, mb, keep=None):
    """rl.py update() with: vLLM sampler logprobs as old_lp (ratio := 1 where the sampler logp is unknown, i.e.
    the re-appended stop token), logits only for the completion positions (logits_to_keep), fp32 vocab math
    in chunks, use_cache off, exact weighted grad sync. Returns rl.py's stats + sampler_abs_dlogp.
    ScaleRL variant: a.loss (ppo | cispo), a.loss_agg (token | seq | prompt) and keep (effective-batch mask from the
    zero-variance filter, None = everything) go through pg_token_loss() / loss_weights(); the legacy flags reproduce the
    original arithmetic bit for bit."""
    import torch
    import torch.distributed as dist
    import rl_hf as R
    from mxf.config import STEER_COEFF
    n, L = ids.shape
    T = L - p_len
    gen_mask = attn[:, p_len:].bool()
    total_tok = max(int(gen_mask.sum()), 1)
    w_all, sync_w = loss_weights(gen_mask, a.loss_agg, a.group_size, keep)
    lo, hi = 1 - a.clip_eps, 1 + a.clip_eps
    trunc_cap = a.cispo_eps_max if a.loss == "cispo" else a.tis_cap
    t_ref = time.time()
    # micro-batches of LENGTH-SORTED rollouts, each padded only to ITS longest sequence: the loss weights are
    # per-sequence and independent of batching, so this is exactly the same gradient as global padding while
    # most micro-batches run at ~p_len+30 instead of p_len+96 tokens
    lens = gen_mask.sum(1)
    order = torch.argsort(lens)

    def chunks(size):
        for s in range(0, n, size):
            ix = order[s : s + size]
            # pad to a multiple of 16 completion tokens (right padding is masked -> numerically inert; fewer distinct
            # shapes -> far fewer fla Triton autotune stalls in the first steps)
            yield ix, p_len + min(T, -(-int(lens[ix].max()) // 16) * 16)
    ref_lp_all = None
    if a.kl_coef > 0:
        ref_lp_all = torch.zeros_like(old_lp)
        actor.set_adapter("ref")
        try:
            with torch.no_grad():
                for ix, Lc in chunks(a.ref_micro_batch):
                    Tc = Lc - p_len
                    b_ids, b_attn = ids[ix, :Lc].to(device), attn[ix, :Lc].to(device)
                    hook = R.make_inject_hook([dirs_rep[i : i + 1] for i in ix.tolist()], [[marker]] * len(ix),
                                              STEER_COEFF, device, torch.bfloat16)
                    with R.hooked(submodule, hook):
                        lg = actor(input_ids=b_ids, attention_mask=b_attn, use_cache=False, logits_to_keep=Tc + 1).logits[:, :-1]
                    for c0 in range(0, Tc, a.vocab_chunk):
                        c1 = min(c0 + a.vocab_chunk, Tc)
                        ref_lp_all[ix, c0:c1] = torch.log_softmax(lg[:, c0:c1].float(), -1).gather(
                            -1, b_ids[:, p_len + c0 : p_len + c1, None]).squeeze(-1).cpu()
                    del lg
        finally:
            actor.set_adapter("default")
    t_ref = time.time() - t_ref
    opt.zero_grad(set_to_none=True)
    loss_sum, clipped_tok, ent_sum, kl_sum, ratio_sum, dlp_sum, dlp_n = 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0
    isw_sum, trunc_tok = 0.0, 0
    t_fb = time.time()
    for ix, Lc in chunks(mb):
        Tc = Lc - p_len
        b_ids, b_attn = ids[ix, :Lc].to(device), attn[ix, :Lc].to(device)
        m = gen_mask[ix, :Tc].to(device); w = w_all[ix, :Tc].to(device); A = adv[ix, None].to(device)
        olp = old_lp[ix, :Tc].to(device); kn = known[ix, :Tc].to(device)
        hook = _hook_outside_autocast(R.make_inject_hook([dirs_rep[i : i + 1] for i in ix.tolist()], [[marker]] * len(ix), STEER_COEFF, device, torch.bfloat16),
                                      a.autocast_bf16)
        with R.hooked(submodule, hook):
            with _policy_precision(actor, a.autocast_bf16):   # --autocast-bf16: bf16 LoRA matmuls/activations; fp32 vocab math below is outside
                logits = actor(input_ids=b_ids, attention_mask=b_attn, use_cache=False, logits_to_keep=Tc + 1).logits[:, :-1]
            new_lp, ent = _chunked_logp(logits, b_ids[:, p_len:], a.vocab_chunk, a.entropy_coef > 0)
            del logits
            olp_eff = torch.where(kn, olp, new_lp.detach())
            loss_tok, ratio, rho = pg_token_loss(new_lp, olp_eff, A, a.loss, a.clip_eps, a.tis_cap, a.cispo_eps_max)
            loss = (loss_tok * w).sum()
            ent_sum += float((ent.detach() * m).sum())
            if a.entropy_coef > 0:
                loss = loss - a.entropy_coef * (ent * w).sum()
            if a.kl_coef > 0:
                ref_lp = ref_lp_all[ix, :Tc].to(device)
                delta = ref_lp - new_lp
                kl = (torch.exp(delta) - delta - 1).clamp(0.0, a.kl_cap)
                loss = loss + a.kl_coef * (kl * w).sum()
                kl_sum += float((kl.detach() * m).sum())
            loss.backward()
        loss_sum += loss.item()
        clipped_tok += int((((ratio < lo) | (ratio > hi)) & m).sum())
        ratio_sum += float((ratio.detach() * m).sum())
        with torch.no_grad():   # the IS weight the gradient actually sees: min(rho, eps_max) (cispo) / ratio where the PPO clip is inactive
            eff = ratio.detach() if a.loss == "cispo" else ratio.detach() * ~(((ratio > hi) & (A > 0)) | ((ratio < lo) & (A < 0)))
            isw_sum += float((eff * m).sum()); trunc_tok += int(((rho > trunc_cap) & m).sum())
        mk = m & kn
        dlp_sum += float(((new_lp.detach() - olp).abs() * mk).sum()); dlp_n += int(mk.sum())
        del new_lp, ent, ratio, loss, olp_eff, loss_tok, rho, eff
    t_fb = time.time() - t_fb
    params = [p for p in actor.parameters() if p.requires_grad]
    t_sync = time.time()
    tot_w = sync_w
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        tot_w = _sync_grads(params, sync_w, a.backend, device)
    t_sync = time.time() - t_sync
    skipped = 0
    if tot_w <= 0:               # zero-variance filter emptied the global effective batch (never with keep=None)
        opt.zero_grad(set_to_none=True); gn = 0.0; skipped = 1
        print("[update] empty effective batch (every group zero-variance) -- skipping step", flush=True)
    else:
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm))
        if math.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad(set_to_none=True); skipped = 1
            print(f"[update] non-finite grad norm ({gn}) -- skipping step", flush=True)
    return {"loss": loss_sum, "grad_norm": gn, "clipfrac": clipped_tok / total_tok, "entropy": ent_sum / total_tok,
            "kl": kl_sum / total_tok, "ratio_mean": ratio_sum / total_tok, "sampler_abs_dlogp": dlp_sum / max(dlp_n, 1),
            "t_ref": t_ref, "t_fb": t_fb, "t_sync": t_sync, "n_unknown_lp": int((gen_mask & ~known).sum()),
            "is_weight_mean": isw_sum / total_tok, "is_trunc_frac": trunc_tok / total_tok, "sync_w": sync_w, "skipped": skipped}


def find_micro_batch(actor, opt, submodule, prompt_ids, marker, a, device, cands, tag):
    """Largest micro-batch whose forward+backward at MAX length (prompt + max_new_tokens) fits with <90% of the GPU
    allocated. Synthetic tokens; same hook, same chunked vocab math as update_disagg.

    Strategy (Sep 3): measure mb=1 and mb=2 (always fit), fit peak ~= fixed + mb * per_seq, predict the largest candidate
    under 85% of the GPU and VERIFY it (<90%); only on a failed verification step down. The old descending scan started at
    mb=64 and OOM'd on purpose -- on the H200:4 run a CUDA OOM mid-forward left the partial graph resident (every later
    candidate saw ~138 GB 'allocated by PyTorch' on a 140 GB card, with only 56.6 GB resident before the scan), so this
    probe never OOMs by design. Each attempt runs in its own function so no local of a failed attempt can outlive it."""
    import torch
    import torch.nn.functional as F
    import rl_hf as R
    from mxf.config import D_MODEL, STEER_COEFF
    L = len(prompt_ids) + a.max_new_tokens
    p_len = len(prompt_ids)
    total = torch.cuda.get_device_properties(0).total_memory
    GB = 2**30
    gc.collect(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    free0, _ = torch.cuda.mem_get_info()
    _log(tag, f"micro-batch probe @ L={L} (prompt {p_len} + {a.max_new_tokens} new): resident {base / GB:.1f} GB, "
              f"free {free0 / GB:.1f} / {total / GB:.0f} GB (device {torch.cuda.current_device()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    def attempt(mb):
        """-> (ok, peak_bytes | None, err | None). All tensors are locals here and die with the frame."""
        ok, peak, err = False, None, None
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(1000, 100000, (mb, L), device=device)
            ids[:, :p_len] = torch.tensor(prompt_ids, device=device)
            attn = torch.ones_like(ids)
            dirs = F.normalize(torch.randn(mb, D_MODEL, device=device), dim=-1)
            hook = _hook_outside_autocast(R.make_inject_hook([dirs[i : i + 1] for i in range(mb)], [[marker]] * mb, STEER_COEFF, device, torch.bfloat16),
                                          getattr(a, "autocast_bf16", False))
            with R.hooked(submodule, hook):
                with _policy_precision(actor, getattr(a, "autocast_bf16", False)):
                    logits = actor(input_ids=ids, attention_mask=attn, use_cache=False, logits_to_keep=L - p_len + 1).logits[:, :-1]
                new_lp, ent = _chunked_logp(logits, ids[:, p_len:], a.vocab_chunk, False)
                del logits
                loss = new_lp.mean() * 0.0
                loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            ok = peak < 0.90 * total
        except torch.cuda.OutOfMemoryError as e:
            ok = False
            err = re.sub(r"\s+", " ", str(e))[:420]
        finally:
            opt.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
        return ok, peak, err

    res, chosen = {}, None

    def record(mb, ok, peak, err, note=""):
        res[mb] = {"ok": ok, "peak_gb": (peak / GB) if peak else None}
        after = torch.cuda.memory_allocated()
        leak = f" | LEAK: {(after - base) / GB:.1f} GB still allocated after the attempt" if after > base + GB else ""
        _log(tag, f"mb {mb}: {'OK' if ok else 'OOM/too tight'}" + (f" peak {peak / GB:.1f} GB / {total / GB:.0f}" if peak else "")
                  + (f" {note}" if note else "") + (f" | {err}" if err else "") + leak)
        return after > base + GB

    ok1, p1, e1 = attempt(1); record(1, ok1, p1, e1, "(calibration)")
    ok2, p2, e2 = attempt(2); leaked = record(2, ok2, p2, e2, "(calibration)")
    if ok1 and ok2 and not leaked:
        per = max(p2 - p1, 64 * 2**20)
        fixed = max(p1 - base - per, 0)
        pred = int((0.85 * total - base - fixed) // per)
        _log(tag, f"probe fit: {per / GB:.2f} GB/seq + {fixed / GB:.1f} GB fixed on {base / GB:.1f} GB resident -> "
                  f"predicted max mb {pred} at 85% of {total / GB:.0f} GB")
        for mb in sorted({c for c in cands if 2 < c <= pred}, reverse=True):
            ok, peak, err = attempt(mb); leaked = record(mb, ok, peak, err, "(verify)")
            if ok:
                chosen = mb
                break
            if leaked:
                _log(tag, "verification OOM leaked GPU memory; not probing further")
                break
        if chosen is None and ok2:
            chosen = 2 if pred >= 2 else 1
    else:
        # calibration itself failed (should not happen with >30 GB free) -- fall back to the old descending scan
        _log(tag, f"calibration failed (ok1={ok1} ok2={ok2} leaked={leaked}); falling back to the descending scan over {cands}")
        for mb in cands:
            ok, peak, err = attempt(mb); record(mb, ok, peak, err)
            if ok:
                chosen = mb
                break
    return chosen, res


def _save_adapter_for_vllm(actor, lora_dir, dtype):
    """rl.py's _save_adapter_for_vllm (module names renamed to the Qwen3_5ForConditionalGeneration layout vLLM
    serves) with a configurable dtype. bf16 halves the write; vLLM casts LoRA weights to the model dtype on
    load, so the served policy is bit-identical either way. Returns (n_tensors, timings)."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    os.makedirs(lora_dir, exist_ok=True)
    t0 = time.time()
    sd = get_peft_model_state_dict(actor, adapter_name="default")
    out = {}
    for k, v in sd.items():
        k2 = k if "language_model" in k else k.replace("model.layers.", "model.language_model.layers.", 1)
        out[k2] = v.detach().to(dtype).to("cpu", copy=True).contiguous()
    torch.cuda.synchronize()
    t1 = time.time()
    save_file(out, f"{lora_dir}/adapter_model.safetensors", metadata={"format": "pt"})
    actor.peft_config["default"].save_pretrained(lora_dir)
    t2 = time.time()
    return len(out), {"state_dict_s": t1 - t0, "write_s": t2 - t1, "bytes": sum(v.numel() * v.element_size() for v in out.values())}


def _write_eval_request(work, k, adapter_step, EV, EX, a, tag):
    """rank 0: everything the rollout ranks need to generate ckpt k's eval texts (~30 MB of unit directions)."""
    import torch
    os.makedirs(f"{work}/eval_req", exist_ok=True)
    req = {"ckpt_step": k, "adapter_step": adapter_step, "t": time.time(), "sets": _eval_sets_from_assets(EV, EX, a)}
    p = _eval_req_path(work, k)
    torch.save(req, p + ".tmp"); os.replace(p + ".tmp", p)
    _log(tag, f"eval request written for ckpt {k} (adapter step {adapter_step}): "
              + ", ".join(f"{st['name']}x{st['n_rows']}x{st['n']}" for st in req["sets"]))


def _score_eval_block(k, shard_paths, EV, EX, IX, actor, tok, device, rank, world, a):
    """ALL trainer ranks: load the rollout ranks' shards, score rows i % world == rank of every set on the CLEAN base
    (exactly rl.py inline_eval's scoring + inline_extra_evals.run_extra_evals_gpu's scoring half), all_gather, and
    on rank 0 reduce to rl.py's eval/* keys (+ extra/locality + adversarial of the previous ckpt's judge texts).
    Returns (ev, ex) dicts on rank 0 ({} elsewhere); errors travel as data so no rank can deadlock."""
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    t0 = time.time()
    EU, es, sae = EV["EU"], EV["es"], EV["sae"]
    pending = IX._broadcast_pending(EX, rank, world) if (EX is not None and IX is not None) else []
    local = {}
    try:
        shards = [torch.load(p, weights_only=False) for p in shard_paths]
        errs = [sh["error"] for sh in shards if sh.get("error")]
        if errs:
            raise RuntimeError("rollout shard errors: " + " | ".join(errs))
        texts = _eval_merge_shards(shards)
        bo = a.eval_bo
        for fam in EV["fams"]:
            du = es[f"{fam}_dirs"]
            rows = [i for i in sorted(texts.get(fam, {})) if i % world == rank]
            if rows:
                flat = [t for i in rows for t in texts[fam][i]]
                rd = F.normalize(torch.stack([du[i] for i in rows for _ in range(bo)]).float(), dim=-1)
                cos = EU.score_probe_cos(flat, rd, actor, tok, device).view(len(rows), bo).max(1).values
                local[fam] = {int(i): float(c) for i, c in zip(rows, cos.tolist())}
            else:
                local[fam] = {}
        feats = EV["feats"]
        rows = [i for i in sorted(texts.get("sae", {})) if i % world == rank]
        if rows:
            flat = [t for i in rows for t in texts["sae"][i]]
            fl = [feats[i] for i in rows for _ in range(bo)]
            acts, peaks = EU.score_sae_peaks(flat, fl, sae, actor, tok, device)
            acts = acts.view(len(rows), bo)
            best, arg = acts.max(1)
            pk = peaks.view(len(rows), bo, -1)[torch.arange(len(rows)), arg]
            local["sae"] = {int(i): float(v) for i, v in zip(rows, best.tolist())}
            local["sae_peak"] = {int(i): pk[j].half().numpy().tobytes() for j, i in enumerate(rows)}
        else:
            local["sae"], local["sae_peak"] = {}, {}
        local["extra"] = None
        if EX is not None and IX is not None:
            xf, subsae, fidx = EX["feats"], EX["subsae"], EX["fidx"]
            n_roll = EX["n_rollouts"]
            xrows = [i for i in sorted(texts.get("extra", {})) if i % world == rank]
            flat_t = [t for i in xrows for t in texts["extra"][i]]
            flat_f = [xf[i] for i in xrows for _ in range(n_roll)]
            loc_rows = IX.locality_rows_from_profiles(IX._profiles(flat_t, flat_f, actor, tok, device, subsae) if flat_t else [], EX["fire"])
            loc = {xf[i]: loc_rows[j * n_roll:(j + 1) * n_roll] for j, i in enumerate(xrows)}
            adv = []
            for item in pending:
                res = {"src": item["src_ckpt_step"], "true": {}, "naive": {}}
                at, af, tags = [], [], []
                for arm in ("true", "naive"):
                    for f, txts in item[arm].items():
                        f = int(f)
                        if f in fidx and fidx[f] % world == rank:
                            for t in txts:
                                at.append(t); af.append(f); tags.append((arm, f))
                profs = IX._profiles(at, af, actor, tok, device, subsae) if at else []
                for (arm, f), pr in zip(tags, profs):
                    res[arm].setdefault(f, []).append(float(pr.max()) if len(pr) else 0.0)
                adv.append(res)
            local["extra"] = {"texts": {xf[i]: texts["extra"][i] for i in xrows}, "loc": loc, "adv": adv}
        local["t_gen"] = float(max(sh.get("t_gen", 0.0) for sh in shards))
    except Exception as e:  # noqa
        local = {"error": f"rank{rank}: {type(e).__name__}: {str(e)[:300]}"}
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank != 0:
        return {}, {}
    errs = [g["error"] for g in gathered if "error" in g]
    if errs:
        return {"error": " | ".join(errs)}, {}
    merged = {}
    for g in gathered:
        for key, d in g.items():
            if key in ("extra", "t_gen"):
                continue
            merged.setdefault(key, {}).update(d)
    out = {}
    for fam in EV["fams"]:
        vals = np.array([merged[fam][i] for i in sorted(merged[fam])], dtype=np.float64)
        out[f"eval/{fam}/cos"] = float(vals.mean())
    idx = sorted(merged["sae"])
    best = np.array([merged["sae"][i] for i in idx], dtype=np.float64)
    cp = EV["cp"][idx]
    na = best / np.maximum(cp, 1e-6)
    out["eval/sae/norm_act"] = float(na.mean())
    if merged.get("sae_peak"):
        peak_h = torch.from_numpy(np.stack([np.frombuffer(merged["sae_peak"][i], dtype=np.float16) for i in idx]).astype(np.float32))
        ranks = EU.sae_rank_at_peaks(sae, peak_h, [EV["feats"][i] for i in idx]).astype(np.float64)
        out["eval/sae/rank1_frac"] = float(np.mean(ranks == 1))
        out["eval/sae/rank_le5"] = float(np.mean(ranks <= 5))
        out["eval/sae/mean_rank"] = float(ranks.mean())
        out["eval/sae/mrr"] = float(np.mean(1.0 / ranks))
    out["eval/sae/fired"] = float(np.mean(best > EU.SAE_FIRE))
    out["eval/sae/beat_corpus"] = float(np.mean(best > cp))
    out["eval/sae/unverbalized_frac"] = float(np.mean(best <= EU.SAE_FIRE))
    out["eval/sae/unverbalized_p10"] = float(np.mean(na < 0.10))
    cos_keys = [kk for kk in out if kk.startswith("eval/") and kk.endswith("/cos") and kk.split("/")[1] not in EU.CONTROL_FAMS and kk.split("/")[1] != "sae"]   # sae/cos is a diagnostic, not a mean_all family
    out["eval/mean_all"] = float(np.mean([out[kk] for kk in cos_keys]))
    for fam in EV["fams"]:
        out[f"eval/all/{fam}_cos"] = out[f"eval/{fam}/cos"]
    out["eval/all/sae_norm_act"] = out["eval/sae/norm_act"]
    out["eval/all/sae_unverbalized"] = out["eval/sae/unverbalized_frac"]
    out["time/inline_eval_s"] = time.time() - t0
    out["time/inline_eval_gen_s"] = float(max(g.get("t_gen", 0.0) for g in gathered))
    ex = {}
    if EX is not None and IX is not None and all(g.get("extra") is not None for g in gathered):
        texts_x, loc = {}, {}
        adv_true, adv_naive = {}, {}
        for g in gathered:
            texts_x.update(g["extra"]["texts"]); loc.update(g["extra"]["loc"])
            for res in g["extra"]["adv"]:
                for arm, store in (("true", adv_true), ("naive", adv_naive)):
                    for f, acts in res[arm].items():
                        store.setdefault(res["src"], {}).setdefault(int(f), []).extend(acts)
        ex = IX.aggregate_locality(loc, EX["corpus_peak"])
        for src in sorted(set(adv_true) | set(adv_naive)):
            m = IX.adversarial_metrics(adv_true.get(src, {}), adv_naive.get(src, {}), EX["corpus_peak"], fire=EX["fire"])
            IX._RESULTS_Q.put((src, m))
        ex["extra/adversarial/n_pending_scored"] = float(len(pending))
        ex["time/extra_eval_gpu_s"] = time.time() - t0
        EX["last_rollouts"] = {"ckpt_step": k, "rollouts": texts_x}
        try:
            os.makedirs(EX["out_dir"], exist_ok=True)
            IX._save_json_atomic({"ckpt_step": k, "rollouts": {str(f): v for f, v in texts_x.items()},
                                  "locality": {str(f): v for f, v in loc.items()}}, os.path.join(EX["out_dir"], f"rollouts_ckpt{k}.json"))
        except Exception as e:  # noqa
            _log("T0", f"could not write extra-eval rollouts artifact: {e}")
    return out, ex


def _publish_adapter(actor, submodule, prompt, marker, device, work, step, keep, tag, fp32=False):
    """rank 0: adapter in vLLM key layout + ||h_marker|| (adapter ON) -> lora/step_<k>, then flip `latest`."""
    import torch
    import rl_hf as R
    t0 = time.time()
    hnorm = R._marker_norm(actor, submodule, prompt, marker, device, adapter=True)
    t_norm = time.time() - t0
    d = f"{work}/lora/step_{step}"
    n, tm = _save_adapter_for_vllm(actor, d, torch.float32 if fp32 else torch.bfloat16)
    json.dump({"step": step, "hnorm": hnorm, "n_tensors": n, "t": time.time()}, open(f"{d}/meta.json", "w"))
    _atomic_write_text(f"{work}/lora/latest", str(step))
    for old in sorted(glob.glob(f"{work}/lora/step_*"), key=lambda p: int(p.rsplit("_", 1)[-1]))[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    tot = time.time() - t0
    if step <= 1:
        _log(tag, f"publish step {step}: hnorm {t_norm:.2f}s | state_dict->cpu {tm['state_dict_s']:.2f}s | write {tm['bytes'] / 2**30:.2f} GB in {tm['write_s']:.2f}s -> {d}")
    return hnorm, tot


def _pick_blocks(work, n_needed, drop_stale):
    files = _queue_files(work)
    if len(files) < n_needed:
        return None, []
    if drop_stale:
        take, stale = files[-n_needed:], files[:-n_needed]
    else:
        take, stale = files[:n_needed], []
    return take, stale


def run_trainer(a):
    import numpy as np
    import torch
    import torch.distributed as dist
    if a.no_fla:
        sys.modules["fla"] = None      # transformers' use_kernel_func_from_hub_with_fallback then keeps the torch GDN path
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import wandb
    import rl_hf as R
    from mxf.config import INJECT_LAYER, MODEL, TrainConfig
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids

    rank = int(os.environ["DISAGG_RANK"]); world = int(os.environ["DISAGG_WORLD"]); is_main = rank == 0
    tag = f"T{rank}"
    work = a.work_dir
    torch.manual_seed(a.seed + rank)
    torch.cuda.set_device(0)
    device = "cuda:0"
    if world > 1:   # nccl for the GPU flat-buffer grad all-reduce, gloo for the CPU-tensor collectives of the eval code
        dist.init_process_group("cpu:gloo,cuda:nccl" if a.backend == "nccl" else "gloo",
                                init_method=f"tcp://127.0.0.1:{a.master_port}", rank=rank, world_size=world)
    try:
        import fla  # noqa
        fla_v = getattr(fla, "__version__", "?")
    except Exception:  # noqa
        fla_v = None
    _log(tag, f"world {world} backend {a.backend} | fla {'v' + str(fla_v) if fla_v else 'ABSENT (torch GDN fallback)'} "
              f"| torch {torch.__version__} | gpu {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.prompt_file:
        _job = open(a.prompt_file).read()
        _txt = tok.apply_chat_template([{"role": "user", "content": _job}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok.encode(_txt, add_special_tokens=False)
        _inj = tok(a.inj_char, add_special_tokens=False).input_ids
        if len(_inj) != 1:
            raise SystemExit("--inj-char %r must be a SINGLE token, got %d: %s"
                             % (a.inj_char, len(_inj), _inj))
        _hits = [i for i, t in enumerate(prompt_ids) if t == _inj[0]]
        if len(_hits) != 1:
            raise SystemExit("--prompt-file must contain EXACTLY one %r (token id %d), found %d. "
                             "The AV was SFT'd with inv_core.INJ_CHAR = U+321C; a different char "
                             "means the injection would land in the wrong place."
                             % (a.inj_char, _inj[0], len(_hits)))
        mpos = [_hits[0]]
        # neighbour check, as inv_train asserts: the marker sits inside <concept>...</concept>, so
        # a shifted position would inject into the tag rather than the slot.
        _lo = tok("<concept>", add_special_tokens=False).input_ids
        _hi = tok("</concept>", add_special_tokens=False).input_ids
        _k = mpos[0]
        if not (prompt_ids[_k - len(_lo):_k] == _lo and
                prompt_ids[_k + 1:_k + 1 + len(_hi)] == _hi):
            _log(tag, "WARNING marker neighbours are not <concept>/</concept>; injection may be "
                      "misplaced relative to SFT")
        _log(tag, "prompt from %s: %d tokens, marker at %d (%d tokens follow it) -- maemm's own "
                  "layout puts the marker last; a mid-prompt marker is reported to weaken "
                  "conditioning" % (a.prompt_file, len(prompt_ids), mpos[0],
                                    len(prompt_ids) - 1 - mpos[0]))
    else:
        prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    B, G = a.groups_per_step, a.group_size
    tr = TrainConfig()

    t0 = time.time()
    actor = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    if a.init_adapter:
        actor = PeftModel.from_pretrained(actor, a.init_adapter, is_trainable=True)
    else:
        actor = get_peft_model(actor, LoraConfig(r=tr.lora_r, lora_alpha=tr.lora_alpha, lora_dropout=0.0, use_rslora=True,
                                                 target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    actor.train()

    AR_REWARD = None
    if a.ar_reward:
        # The modulation-lens reward gets its OWN backbone truncated to read_layer+1: the policy
        # needs every layer to generate, but the AR must be read on the truncation it was trained
        # with -- the same adapter read through the full 64-layer stack HALVES the reward
        # (0.331 vs 0.759, measured on identical atoms and activations). ~38 GB bf16 on top of
        # the actor, which is why it is opt-in.
        import ar_reward as ARR
        for _n, _v in (("--ar-jlens", a.ar_jlens), ("--ar-affine", a.ar_affine)):
            if not _v:
                raise SystemExit("%s is required with --ar-reward" % _n)
        AR_REWARD = ARR.ARReward(a.ar_reward, a.ar_jlens, a.ar_affine, device=device,
                                 read_layer=getattr(a, "layer", 42),
                                 max_tokens=a.bullet_max_tok, amu_path=a.ar_amu)
        if a.ar_whiten:
            AR_REWARD.load_whitener(a.ar_whiten, a.ar_whiten_key)
            _log(tag, "reward whitened in J-space: %s[%s]" % (a.ar_whiten, a.ar_whiten_key))
        AR_REWARD.build_own(MODEL)
        # The distinct-token fraction needs token ids only. Skip the clean-base logits forward
        # unless a fluency floor is actually in play, so a distinct-only gate is free.
        AR_REWARD.need_logp = a.fluency_floor is not None
        _log(tag, "AR reward live: bullets=%d max_tok=%d affine=%s amu=%s | gates flu=%s dis=%s"
                  % (a.bullets, a.bullet_max_tok, bool(a.ar_affine), bool(a.ar_amu),
                     a.fluency_floor, a.distinct_floor))
    if a.fp32_head:   # before the micro-batch search so its +memory is part of the OOM probe
        install_fp32_head(actor)
        _log(tag, "lm_head recomputed in fp32 (ScaleRL precision fix, trainer side; the vLLM sampler stays bf16-head/fp32-softmax)")
    opt = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0,
                            eps=a.adam_eps, betas=tuple(a.adam_betas))
    optim_p = os.path.join(a.init_adapter or "", "optim.pt")
    if a.init_adapter and os.path.exists(optim_p):
        opt.load_state_dict(torch.load(optim_p, map_location="cpu"))
        _log(tag, f"AdamW state restored from {optim_p}")
    submodule = get_layer(actor, INJECT_LAYER)
    if a.kl_coef > 0:
        ref_src = a.ref_adapter or a.init_adapter
        assert ref_src, "--kl-coef needs --init-adapter or --ref-adapter"
        actor.load_adapter(ref_src, adapter_name="ref")
        actor.set_adapter("default")
    n_train = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    _log(tag, f"actor ready in {time.time() - t0:.0f}s | trainable {n_train / 1e6:.0f}M | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB")

    # publish the init policy FIRST so the rollout ranks start generating while we tune the micro-batch
    if is_main:
        for d in ("lora", "queue"):
            os.makedirs(f"{work}/{d}", exist_ok=True)
        hnorm0, t_pub = _publish_adapter(actor, submodule, prompt, marker, device, work, a.step_offset, a.keep_loras, tag, a.publish_fp32)
        _log(tag, f"published init adapter as step {a.step_offset} (||h_marker|| = {hnorm0:.2f}) in {t_pub:.1f}s")

    mb = a.micro_batch
    mb_res = {}
    if mb <= 0:
        cands = [int(x) for x in a.mb_candidates.split(",")]
        mb, mb_res = find_micro_batch(actor, opt, submodule, prompt_ids, marker, a, device, cands, tag)
        assert mb is not None, f"no micro-batch candidate fits: {mb_res}"
        if world > 1:
            t = torch.tensor([mb], dtype=torch.int64, device=device if a.backend == "nccl" else "cpu")
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            mb = int(t.item())
    _peak = (mb_res.get(mb) or {}).get("peak_gb")
    _log(tag, f"micro-batch = {mb} (no gradient checkpointing; autocast bf16 policy forward {'ON' if a.autocast_bf16 else 'off'}"
              + (f"; probe peak {_peak:.1f} GB" if _peak else "") + ")")
    # ---- inline eval assets (held-out sets + SAE on every trainer rank; extra-eval testbed/judge) — after the
    # micro-batch search so the SAE's 2.7 GB is not part of the OOM probe ----
    EV = R.load_eval_assets(a, device, is_main) if a.inline_eval_every > 0 else None
    EX, IX = None, None
    if EV is not None and not a.no_extra_evals:
        try:
            import inline_extra_evals as IX
            EX = IX.prepare_extra_eval_assets(a, device, rank, world, is_main, sae=EV["sae"])
        except Exception as e:  # noqa
            EX, IX = None, None
            if is_main:
                _log(tag, f"extra-eval DISABLED ({type(e).__name__}: {e})")
    if world > 1:   # every rank must agree on whether EV exists (a failed load on one rank would otherwise desync the collectives)
        t = torch.tensor([0 if EV is None else 1], dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        if int(t.item()) == 0:
            EV, EX = None, None
    if is_main and EV is not None:
        _log(tag, f"inline eval every {a.inline_eval_every} steps: rollout ranks generate {len(EV['fams'])} families x {len(EV['es'][EV['fams'][0] + '_dirs'])} "
                  f"dirs x Bo{a.eval_bo} + sae + {'extra testbed' if EX is not None else 'NO extra evals'}; trainer ranks score sharded")

    adv_mode = a.adv_mode or ("batch" if a.batch_norm else ("group" if a.std_norm else "none"))
    use_gates = a.fluency_floor is not None or a.distinct_floor is not None
    tgt_map = {}
    if is_main and a.transcript_every > 0 and a.direction_source == "cluster" and os.path.exists(f"{a.data_dir}/records.jsonl"):
        with open(f"{a.data_dir}/records.jsonl") as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line)
                except Exception:  # noqa
                    continue
                vi = rec.get("vec_idx", i)
                if vi not in tgt_map and rec.get("target_text") is not None:
                    tgt_map[vi] = (rec.get("family"), rec["target_text"][:240])
    npr, npr_path, n_bank_avail = None, f"{work}/npr/dropped.json", 0
    if a.npr_threshold > 0 and a.direction_source == "cluster":
        npr = NPRTracker(a.npr_threshold, a.npr_pass_cos)          # rank 0 drives it; every rank sees the same gathered rewards
        _, _nv, _er = _bank_open(a)
        n_bank_avail = int(_nv - _er)
    if is_main:
        _log(tag, f"B={B} groups x G={G} per step from {a.blocks_per_step} block(s) of {a.rollout_block_groups} | adv {adv_mode} "
                  f"| gates {use_gates} | mb {mb} | ref-mb {a.ref_micro_batch} | queue cap {a.max_queue_blocks} | {'drop-stale' if a.drop_stale else 'FIFO'}")
        _log(tag, f"recipe {a.recipe or 'legacy'} | loss {a.loss} " + (f"eps_max {a.cispo_eps_max}" if a.loss == "cispo" else f"clip {a.clip_eps} tis {a.tis_cap}")
                  + f" | loss-agg {a.loss_agg} | zero-var filter {a.zero_var_filter} (eps {a.zero_var_eps}) | NPR {a.npr_threshold}"
                  + (f" (pass cos {a.npr_pass_cos}, {n_bank_avail} directions)" if npr is not None else " (off)")
                  + f" | max lag {a.max_lag} step(s) | fp32 head {a.fp32_head} | length control {a.length_control}")
        if not a.no_wandb:
            wandb.init(project="maxact-fast", name=a.run_name, config={**vars(a), "micro_batch_used": mb, "mb_search": mb_res,
                                                                        "fla": fla_v, "disagg": True},
                       id=a.wandb_id or None, resume="must" if a.wandb_id else None)
            wandb.define_metric("ckpt_step")
            wandb.define_metric("eval/*", step_metric="ckpt_step")
            wandb.define_metric("extra/*", step_metric="ckpt_step")
        os.makedirs(a.save_dir, exist_ok=True)
        json.dump({"micro_batch": mb, "search": mb_res}, open(f"{work}/trainer_mb.json", "w"))
    eos_set = set(R._eos_ids(tok, actor))
    step_hist = []

    for step in range(a.step_offset, a.total_steps):
        t0 = time.time()
        # ---- rank 0 picks the blocks; everyone loads them (shared /tmp) and takes its whole-group shard ----
        pick = [None, None, None]
        if is_main:
            ev_ready = None
            if EV is not None:   # a complete eval block (all rollout shards present) is scored at the START of the step
                for kk in _eval_requests(work):
                    if _eval_shards_ready(work, kk, a.n_rollout):
                        ev_ready = (kk, [_eval_shard_path(work, kk, r) for r in range(a.n_rollout)])
                        break
                    if time.time() - os.path.getmtime(_eval_req_path(work, kk)) > a.eval_drop_after_s:
                        _log(tag, f"inline-eval ckpt {kk}: shards never completed after {a.eval_drop_after_s:.0f}s — dropped")
                        os.remove(_eval_req_path(work, kk))
            while True:
                take, stale = _pick_blocks(work, a.blocks_per_step, a.drop_stale)
                if take is not None:
                    break
                time.sleep(0.2)
            pick = [take, stale, ev_ready]
        if world > 1:
            dist.broadcast_object_list(pick, src=0)
        take, stale, ev_ready = pick
        if ev_ready is not None:
            kk, paths = ev_ready
            ev, ex = _score_eval_block(kk, paths, EV, EX, IX, actor, tok, device, rank, world, a)
            if is_main:
                for pth in paths + [_eval_req_path(work, kk)]:
                    try:
                        os.remove(pth)
                    except FileNotFoundError:
                        pass
                if "error" in ev:
                    _log(tag, f"inline-eval FAILED for ckpt {kk}: {ev['error']}")
                else:
                    print(f"  [inline-eval] ckpt {kk}: mean_all {ev['eval/mean_all']:.4f} | sae norm_act {ev['eval/sae/norm_act']:.4f} "
                          f"unverb {ev['eval/sae/unverbalized_frac']:.3f} rank1 {ev.get('eval/sae/rank1_frac', float('nan')):.3f} | realact "
                          f"{ev.get('eval/realact/cos', float('nan')):.4f} | scored in {ev['time/inline_eval_s']:.0f}s (rollout gen {ev['time/inline_eval_gen_s']:.0f}s)"
                          + (f" | locality win5 {ex.get('extra/locality/win5_share', float('nan')):.3f} fire {ex.get('extra/locality/fire_frac', float('nan')):.3f}" if ex else ""),
                          flush=True)
                    if not a.no_wandb:
                        wandb.log({**ev, **ex, "ckpt_step": kk})
                    if ex and IX is not None:
                        try:
                            IX.launch_judge_stage(None, kk, EX, a)
                        except Exception as e:  # noqa
                            _log(tag, f"judge launch failed: {type(e).__name__}: {e}")
        t_wait = time.time() - t0
        blocks = [torch.load(f, weights_only=False) for f in take]
        if world > 1:
            dist.barrier()
        if is_main:
            for f in take + stale:
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass
        dirs_all = torch.cat([b["dirs"] for b in blocks])                       # [B, d]
        idx_all = np.concatenate([np.asarray(b["dir_idx"]) for b in blocks])
        gen_all = [g for b in blocks for g in b["gen_ids"]]
        lps_all = [l for b in blocks for l in b["lps"]]
        assert dirs_all.shape[0] == B and len(gen_all) == B * G, f"batch shape {dirs_all.shape[0]} groups / {len(gen_all)} seqs"
        lag = float(np.mean([step - b["adapter_step"] for b in blocks]))
        lag_max = float(max(step - b["adapter_step"] for b in blocks))
        gen_s = float(np.mean([b["gen_s"] for b in blocks]))
        blk_tok = float(np.sum([b["n_tok"] for b in blocks]))
        my_groups = np.array_split(np.arange(B), world)[rank]
        g0, g1 = int(my_groups[0]), int(my_groups[-1]) + 1
        Bl = g1 - g0
        dirs = dirs_all[g0:g1]
        idx = idx_all[g0:g1]
        gen_ids = gen_all[g0 * G : g1 * G]
        lps = lps_all[g0 * G : g1 * G]
        texts = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
        dirs_rep = dirs.repeat_interleave(G, 0).to(device)

        # ---- reward + shaping (identical to rl.py main) ----
        # The modulation-lens reward replaces R.score wholesale; everything downstream
        # (CISPO, batch advantage normalisation, zero-variance filter, NPR, length penalty,
        # truncation reward) is objective-agnostic and untouched.

        t_sc = time.time()
        if use_gates:
            if AR_REWARD is not None:
                r, flu, dis = AR_REWARD.score(texts, dirs_rep, actor, tok, k=a.bullets,
                                              max_tok=a.bullet_max_tok, with_fluency=True,
                                              contrast_negatives=a.reward_contrast_negatives,
                                              contrast_weight=a.reward_contrast_weight,
                                              group_stride=G)
            else:
                r, flu, dis = R.score(texts, dirs_rep, actor, tok, device, a, with_fluency=True)
        else:
            r = (AR_REWARD.score(texts, dirs_rep, actor, tok, k=a.bullets,
                                 max_tok=a.bullet_max_tok,
                                 contrast_negatives=a.reward_contrast_negatives,
                                 contrast_weight=a.reward_contrast_weight,
                                 group_stride=G)
                 if AR_REWARD is not None
                 else R.score(texts, dirs_rep, actor, tok, device, a))
        r = r * a.reward_scale
        raw_r, gate_frac = r.clone(), 1.0                      # raw_r = the TRUE cosine (logged/transcripts), before any shaping (rl.py fd2d144)
        trunc = torch.tensor([len(g) >= a.max_new_tokens and (not g or g[-1] not in eos_set) for g in gen_ids])
        trunc_frac = trunc.float().mean().item()
        if a.trunc_reward is not None and bool(trunc.any()):   # EasyNLA: cap-hit rollouts score a fixed failure reward and still train
            r[trunc] = a.trunc_reward
        gate = torch.ones(Bl * G, dtype=torch.bool)
        if use_gates:
            if a.fluency_floor is not None:
                gate &= flu >= a.fluency_floor
            if a.distinct_floor is not None:
                gate &= dis >= a.distinct_floor
            r = r - a.gate_penalty * (~gate).float()
            gate_frac = gate.float().mean().item()
        if is_main and AR_REWARD is not None and step % 10 == 0:
            _ls = getattr(AR_REWARD, "last_stats", {}) or {}
            _log(tag, "reward terms | scored %s of %d | bullets %.2f | frac_empty %.3f | "
                      "matched_fit %.4f | neg_fit %.4f"
                      % (_ls.get("n_scored"), len(texts), _ls.get("mean_bullets", float("nan")),
                         _ls.get("frac_empty", float("nan")), _ls.get("mean_matched_fit", float("nan")),
                         _ls.get("mean_neg_fit", float("nan"))))
        flu_pct = None
        if (is_main and a.flu_monitor_every > 0 and step % a.flu_monitor_every == 0
                and AR_REWARD is not None):
            # Measure the gate inputs WITHOUT gating on them, so any future floor is set from the
            # observed distribution. Subsampled because it costs a clean-base forward.
            _rs = np.random.default_rng(step)
            _sel = _rs.choice(len(texts), size=min(a.flu_monitor_n, len(texts)), replace=False)
            _f, _d = AR_REWARD.fluency([texts[int(i)] for i in _sel], actor, tok, need_logp=True)
            _fq = np.percentile(_f.numpy(), [1, 5, 10, 25, 50, 90])
            _dq = np.percentile(_d.numpy(), [1, 5, 10, 25, 50, 90])
            flu_pct = {"flu/p%d" % q: float(v) for q, v in zip([1, 5, 10, 25, 50, 90], _fq)}
            flu_pct.update({"dis/p%d" % q: float(v) for q, v in zip([1, 5, 10, 25, 50, 90], _dq)})
            _log(tag, "gate-input percentiles (n=%d) | logp p1/p5/p10/p25/p50/p90 %s | distinct %s"
                      % (len(_sel), " ".join("%.2f" % v for v in _fq),
                         " ".join("%.3f" % v for v in _dq)))
        if a.len_penalty_start is not None:
            over = torch.tensor([max(0, len(g) - a.len_penalty_start) for g in gen_ids], dtype=torch.float32)
            r = r - a.len_penalty_per_tok * over * gate.float()
        adv, keep = compute_advantages_disagg(r, Bl, G, adv_mode, a.zero_var_eps, a.zero_var_filter)
        t_sc = time.time() - t_sc

        _table = None
        if is_main and a.transcript_every > 0 and step % a.transcript_every == 0:
            rows_t = []
            for g in range(min(a.transcript_groups, Bl)):
                vi = int(idx[g])
                fam, tgt = tgt_map.get(vi, (None, None))
                for j in range(min(a.transcript_samples, G)):
                    i = g * G + j
                    rows_t.append({"step": step, "group": g, "vec_idx": vi, "family": fam, "target": tgt, "text": texts[i],
                                   "cos": float(raw_r[i]) / a.reward_scale, "reward": float(r[i]), "adv": float(adv[i]), "n_tok": len(gen_ids[i])})
            with open(f"{a.save_dir}/transcripts.jsonl", "a") as f:
                for row in rows_t:
                    f.write(json.dumps(row) + "\n")
            if not a.no_wandb:
                _table = wandb.Table(columns=list(rows_t[0].keys()), data=[list(x.values()) for x in rows_t])

        # ---- pad + update (old_lp = the SAMPLER's logprobs) ----
        L = p_len + max(len(g) for g in gen_ids)
        ids = torch.full((Bl * G, L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((Bl * G, L), dtype=torch.long)
        old_lp = torch.zeros((Bl * G, L - p_len))
        known = torch.zeros((Bl * G, L - p_len), dtype=torch.bool)
        pt = torch.tensor(prompt_ids, dtype=torch.long)
        for i, (g, lp) in enumerate(zip(gen_ids, lps)):
            ids[i, :p_len] = pt
            ids[i, p_len : p_len + len(g)] = torch.tensor(g)
            attn[i, : p_len + len(g)] = 1
            for j, v in enumerate(lp):
                if v is not None:
                    old_lp[i, j] = float(v); known[i, j] = True
        t_up = time.time()
        if a.warmup_steps > 0:   # linear LR warmup over the first N global steps (stability)
            for _g in opt.param_groups:
                _g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup_steps)

        stats = update_disagg(actor, opt, submodule, ids, attn, p_len, marker, old_lp, known, adv, dirs_rep, a, device, mb, keep=keep)
        if a.entropy_target > 0:   # SAC-style temperature adaptation on the measured per-token entropy of this step
            a.entropy_coef = float(min(a.entropy_coef_max, max(a.entropy_coef_min,
                                   a.entropy_coef * math.exp(a.entropy_adapt_rate * (a.entropy_target - stats["entropy"])))))
        t_up = time.time() - t_up

        # ---- publish the new policy for the rollout ranks ----
        t_pub, hnorm = 0.0, float("nan")
        if is_main and a.publish_every > 0 and (step + 1) % a.publish_every == 0:
            hnorm, t_pub = _publish_adapter(actor, submodule, prompt, marker, device, work, step + 1, a.keep_loras, tag, a.publish_fp32)
            # rl.py evaluates ckpt `step` (the weights after this update) with the adapter published right here
            if EV is not None and step % a.inline_eval_every == 0:
                _write_eval_request(work, step, step + 1, EV, EX, a, tag)

        # ---- logging (reward stats over ALL trainer ranks; uneven shards -> weighted) ----
        secs = time.time() - t0
        mem_alloc = torch.cuda.memory_allocated(device) / 2**30
        mem_peak = torch.cuda.max_memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        n_gen = float(sum(len(g) for g in gen_ids))
        _rg = raw_r.view(Bl, G)
        n_dropped_g = 0.0 if keep is None else float((~keep.view(Bl, G)[:, 0]).sum())
        loc = torch.tensor([n_gen, gate_frac * Bl, r.view(Bl, G).std(1).sum().item(), Bl,
                            _rg.std(1).sum().item(), (_rg.std(1) < 1e-6).float().sum().item(),
                            adv.pow(2).sum().item(), adv.abs().sum().item(), float(Bl * G),
                            stats["n_unknown_lp"], n_dropped_g], dtype=torch.float64)
        gmin, gmax = torch.tensor([_rg.std(1).min().item()]), torch.tensor([_rg.std(1).max().item()])
        if world > 1:
            dev = device if a.backend == "nccl" else "cpu"
            raw_list = [None] * world; r_list = [None] * world; gm_list = [None] * world
            dist.all_gather_object(raw_list, raw_r); dist.all_gather_object(r_list, r); dist.all_gather_object(gm_list, _rg.mean(1))
            raw_r_all, r_all, gmeans = torch.cat(raw_list), torch.cat(r_list), torch.cat(gm_list)
            loc = loc.to(dev); dist.all_reduce(loc); loc = loc.cpu()
            gmin, gmax = gmin.to(dev), gmax.to(dev)
            dist.all_reduce(gmin, op=dist.ReduceOp.MIN); dist.all_reduce(gmax, op=dist.ReduceOp.MAX)
            gmin, gmax = gmin.cpu(), gmax.cpu()
        else:
            raw_r_all, r_all, gmeans = raw_r, r, _rg.mean(1)
        n_gen_all, n_groups_all, n_seq_all = float(loc[0]), float(loc[3]), float(loc[8])
        log = {"reward/mean": raw_r_all.mean().item(), "reward/std": raw_r_all.std().item(), "reward/max": raw_r_all.max().item(),
               "reward/shaped_mean": r_all.mean().item(), "reward/within_group_std": float(loc[2] / n_groups_all),
               "reward/gate_frac": float(loc[1] / n_groups_all), "reward/trunc_frac": trunc_frac,
               **(flu_pct or {}),
               "ratio/clipfrac": stats["clipfrac"], "ratio/mean": stats["ratio_mean"],
               "policy/entropy": stats["entropy"], "policy/kl_to_init": stats["kl"], "policy/entropy_coef": a.entropy_coef,
               "policy/sampler_abs_dlogp": stats["sampler_abs_dlogp"], "policy/offpolicy_lag_steps": lag,
               "loss": stats["loss"], "grad_norm": stats["grad_norm"], "grad_norm_did_clip": float(stats["grad_norm"] > a.max_grad_norm),
               "rollout/mean_logp": float(old_lp[known].mean()) if bool(known.any()) else float("nan"),
               "rollout/len_mean": n_gen_all / (B * G), "tokens_per_sec": n_gen_all / secs,
               "rollout/gen_s": gen_s, "rollout/tok_per_s_per_replica": blk_tok / max(gen_s * len(blocks), 1e-6),
               "rollout/appended_stop": float(sum(b["appended"] for b in blocks)), "rollout/marker_hnorm": hnorm,
               "rollout/queue_depth": float(len(_queue_files(work))), "rollout/blocks_dropped": float(len(stale)),
               "rollout/unknown_lp_tokens": float(loc[9]),
               "time/step_s": secs, "time/wait_rollouts_s": t_wait, "time/score_s": t_sc, "time/update_s": t_up,
               "time/ref_pass_s": stats["t_ref"], "time/fwd_bwd_s": stats["t_fb"], "time/grad_sync_s": stats["t_sync"],
               "time/publish_s": t_pub, "time/rollout_s": gen_s,
               "mem/hf_alloc_gb": mem_alloc, "mem/hf_peak_gb": mem_peak, "micro_batch": mb,
               # ScaleRL diagnostics (present in every run; zero/inert when the variant flags are off)
               "scalerl/is_weight_mean": stats["is_weight_mean"], "scalerl/is_trunc_frac": stats["is_trunc_frac"],
               "scalerl/zero_var_dropped_frac": float(loc[10] / n_groups_all), "scalerl/effective_groups": float(n_groups_all - loc[10]),
               "scalerl/lag_max": lag_max, "scalerl/trunc_frac": trunc_frac, "scalerl/step_skipped": float(stats["skipped"])}
        if npr is not None:   # No-Positive-Resampling bookkeeping on the whole batch (every rank has the gathered rewards; rank 0 publishes)
            log.update(npr.update(idx_all, (raw_r_all / a.reward_scale).numpy(), G))
            log["scalerl/npr_dropped_frac_of_bank"] = len(npr.dropped) / max(n_bank_avail, 1)
            if is_main:
                npr.publish(npr_path)
        if R.SCORE_STATS.get("peak_dist"):
            _pd = torch.cat(R.SCORE_STATS["peak_dist"]); log["reward/peak_dist_mean"] = _pd.mean().item()
            log["reward/peak_in_last5_frac"] = (_pd <= 4).float().mean().item()
        w_std = float(loc[4] / n_groups_all)
        b_std = float(gmeans.std().item()) if len(gmeans) > 1 else 0.0
        log.update({"var/within_group_std_raw": w_std, "var/between_group_std_raw": b_std,
                    "var/zero_var_group_frac": float(loc[5] / n_groups_all),
                    "var/adv_std": float(math.sqrt(max(loc[6] / n_seq_all, 0.0))),   # advantages are zero-mean per group -> std = rms
                    "var/adv_abs_mean": float(loc[7] / n_seq_all),
                    "var/group_std_min": float(gmin), "var/group_std_max": float(gmax), "var/signal_ratio": w_std / (b_std + 1e-9)})
        if _table is not None:
            log["rollouts/samples"] = _table
        step_hist.append({"step": step, "step_s": secs, "wait_s": t_wait, "update_s": t_up, "score_s": t_sc, "lag": lag,
                          "gen_s": gen_s, "reward": log["reward/mean"], "entropy": log["policy/entropy"], "ratio": log["ratio/mean"],
                          "clipfrac": log["ratio/clipfrac"], "dlogp": log["policy/sampler_abs_dlogp"], "len": log["rollout/len_mean"],
                          "peak_gb": mem_peak, "t_ref": stats["t_ref"], "t_fb": stats["t_fb"], "t_sync": stats["t_sync"], "t_pub": t_pub,
                          "n_local": Bl * G})
        if is_main:
            print(f"step {step:05d} | r {log['reward/mean']:.3f} (max {log['reward/max']:.2f}) | ent {log['policy/entropy']:.2f} "
                  f"| ratio {log['ratio/mean']:.3f} clip {log['ratio/clipfrac']:.2%} |dlogp| {log['policy/sampler_abs_dlogp']:.4f} lag {lag:.1f} "
                  f"| kl {log['policy/kl_to_init']:.4f} | len {log['rollout/len_mean']:.0f} | gnorm {log['grad_norm']:.3f} "
                  f"| {secs:.0f}s (wait {t_wait:.0f} score {t_sc:.0f} update {t_up:.0f} [ref {stats['t_ref']:.0f} fb {stats['t_fb']:.0f} sync {stats['t_sync']:.1f}] pub {t_pub:.0f}) "
                  f"| gen {gen_s:.0f}s | peak {mem_peak:.0f}G | queue {int(log['rollout/queue_depth'])}", flush=True)
            if step % 10 == 0:
                print(f"  sample r={raw_r[0]:.2f}: {texts[0][:110]!r}", flush=True)
            if not a.no_wandb:
                wandb.log(log, step=step)
            if IX is not None and EX is not None:
                for cs, m in IX.poll_judge_results():          # judge results arrive minutes later; x-axis = ckpt_step
                    print(f"  [extra-eval] judge results for ckpt {cs}: " + " ".join(
                        f"{kk.split('/')[-1]}={v:.3f}" for kk, v in m.items() if kk.startswith("extra/") and "auc" in kk), flush=True)
                    if not a.no_wandb:
                        wandb.log({**m, "ckpt_step": cs})
            json.dump(step_hist, open(f"{work}/trainer_steps.json", "w"))
            _save_steps = {int(x) for x in a.save_steps.split(",") if x.strip()}
            if (a.save_every and step and step % a.save_every == 0) or (step in _save_steps):
                actor.save_pretrained(f"{a.save_dir}/step_{step}")
                torch.save(opt.state_dict(), f"{a.save_dir}/step_{step}/optim.pt")
                if a.save_every:   # rolling optim.pt cleanup only for the periodic schedule (log-spaced ckpts keep theirs)
                    stale_o = os.path.join(a.save_dir, f"step_{step - 2 * a.save_every}", "optim.pt")
                    if os.path.exists(stale_o) and (step - 2 * a.save_every) not in _save_steps:
                        os.remove(stale_o)
    if is_main:
        actor.save_pretrained(f"{a.save_dir}/final")
        if a.save_every:
            torch.save(opt.state_dict(), f"{a.save_dir}/final/optim.pt")
        # rl.py-format round trip: the checkpoint must load as a PEFT adapter next to the live one
        try:
            actor.load_adapter(f"{a.save_dir}/final", adapter_name="chk_final")
            actor.set_adapter("default")
            _log(tag, f"checkpoint {a.save_dir}/final loads via PeftModel.load_adapter: OK")
        except Exception as e:  # noqa
            _log(tag, f"checkpoint load-back FAILED: {type(e).__name__}: {e}")
        if IX is not None and EX is not None:
            IX.wait_for_judge_stages(900)
            for cs, m in IX.poll_judge_results():
                if not a.no_wandb:
                    wandb.log({**m, "ckpt_step": cs})
        _atomic_write_text(f"{work}/STOP", "done")
        print("RL_DONE", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def run_bench_trainer(a):
    """Trainer-side cost model: micro-batch search, then score()+update_disagg() wall time for several
    rollouts-per-rank sizes on SYNTHETIC rollouts at realistic lengths (uniform 8..96 gen tokens, mean ~40),
    on world = n_trainer ranks (so grad sync is included). Writes <work>/bench_trainer_r<rank>.json."""
    # The bench measures TRAINER cost on synthetic rollouts; it does not build the AR reward (that
    # would add a second ~38 GB backbone and change what is being timed). Refuse rather than score
    # synthetic rollouts with a different objective than the caller asked for -- a bench that
    # silently measures the wrong reward is worse than one that stops.
    AR_REWARD = None
    if a.ar_reward:
        raise SystemExit("--ar-reward is not wired into run_bench_trainer. Bench with the maemm "
                         "objective (drop --ar-reward) or time the real trainer.")
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    if a.no_fla:
        sys.modules["fla"] = None
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import rl_hf as R
    from mxf.config import D_MODEL, INJECT_LAYER, MODEL
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids
    rank = int(os.environ["DISAGG_RANK"]); world = int(os.environ["DISAGG_WORLD"]); tag = f"BT{rank}"
    torch.cuda.set_device(0); device = "cuda:0"
    if world > 1:
        dist.init_process_group(a.backend, init_method=f"tcp://127.0.0.1:{a.master_port}", rank=rank, world_size=world)
    try:
        import fla  # noqa
        fla_v = getattr(fla, "__version__", "?")
    except Exception:  # noqa
        fla_v = None
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.prompt_file:
        _job = open(a.prompt_file).read()
        _txt = tok.apply_chat_template([{"role": "user", "content": _job}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok.encode(_txt, add_special_tokens=False)
        _inj = tok(a.inj_char, add_special_tokens=False).input_ids
        if len(_inj) != 1:
            raise SystemExit("--inj-char %r must be a SINGLE token, got %d: %s"
                             % (a.inj_char, len(_inj), _inj))
        _hits = [i for i, t in enumerate(prompt_ids) if t == _inj[0]]
        if len(_hits) != 1:
            raise SystemExit("--prompt-file must contain EXACTLY one %r (token id %d), found %d. "
                             "The AV was SFT'd with inv_core.INJ_CHAR = U+321C; a different char "
                             "means the injection would land in the wrong place."
                             % (a.inj_char, _inj[0], len(_hits)))
        mpos = [_hits[0]]
        # neighbour check, as inv_train asserts: the marker sits inside <concept>...</concept>, so
        # a shifted position would inject into the tag rather than the slot.
        _lo = tok("<concept>", add_special_tokens=False).input_ids
        _hi = tok("</concept>", add_special_tokens=False).input_ids
        _k = mpos[0]
        if not (prompt_ids[_k - len(_lo):_k] == _lo and
                prompt_ids[_k + 1:_k + 1 + len(_hi)] == _hi):
            _log(tag, "WARNING marker neighbours are not <concept>/</concept>; injection may be "
                      "misplaced relative to SFT")
        _log(tag, "prompt from %s: %d tokens, marker at %d (%d tokens follow it) -- maemm's own "
                  "layout puts the marker last; a mid-prompt marker is reported to weaken "
                  "conditioning" % (a.prompt_file, len(prompt_ids), mpos[0],
                                    len(prompt_ids) - 1 - mpos[0]))
    else:
        prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    t0 = time.time()
    actor = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    actor = PeftModel.from_pretrained(actor, a.init_adapter, is_trainable=True)
    actor.train()
    if a.fp32_head:
        install_fp32_head(actor)
    opt = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0, eps=a.adam_eps, betas=tuple(a.adam_betas))
    submodule = get_layer(actor, INJECT_LAYER)
    if a.kl_coef > 0:
        actor.load_adapter(a.ref_adapter or a.init_adapter, adapter_name="ref"); actor.set_adapter("default")
    resident = torch.cuda.memory_allocated() / 2**30
    _log(tag, f"actor in {time.time() - t0:.0f}s | resident {resident:.1f} GB | fla {fla_v} | world {world} {a.backend}")
    out = {"fla": fla_v, "resident_gb": resident, "gpu": torch.cuda.get_device_name(0), "world": world, "backend": a.backend}
    mb = a.micro_batch
    if mb <= 0:
        mb, res = find_micro_batch(actor, opt, submodule, prompt_ids, marker, a, device, [int(x) for x in a.mb_candidates.split(",")], tag)
        out["mb_search"] = res
        if world > 1:
            t = torch.tensor([mb], dtype=torch.int64, device=device if a.backend == "nccl" else "cpu")
            dist.all_reduce(t, op=dist.ReduceOp.MIN); mb = int(t.item())
    out["micro_batch"] = mb
    t_pub = time.time()
    hn = R._marker_norm(actor, submodule, prompt, marker, device, adapter=True)
    _, tm32 = _save_adapter_for_vllm(actor, f"{a.work_dir}/bench_pub32", torch.float32)
    _, tm16 = _save_adapter_for_vllm(actor, f"{a.work_dir}/bench_pub16", torch.bfloat16)
    out["publish_s"] = time.time() - t_pub; out["hnorm"] = hn; out["publish_fp32"] = tm32; out["publish_bf16"] = tm16
    _log(tag, f"mb {mb} | publish fp32 {tm32} | bf16 {tm16}")
    rng = np.random.default_rng(0)
    G = a.group_size
    rows = []
    for n_roll in [int(x) for x in a.bench_rollouts_per_rank.split(",")]:
        Bl = max(1, n_roll // G)
        lens = rng.integers(a.min_new_tokens, a.max_new_tokens + 1, size=Bl * G)
        gen_ids = [list(rng.integers(1000, 100000, size=int(l))) + [tok.eos_token_id] for l in lens]
        texts = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
        dirs = F.normalize(torch.randn(Bl, D_MODEL), dim=-1)
        dirs_rep = dirs.repeat_interleave(G, 0).to(device)
        torch.cuda.synchronize(); t_s = time.time()
        r = (AR_REWARD.score(texts, dirs_rep, actor, tok, k=a.bullets,
                             max_tok=a.bullet_max_tok)
             if AR_REWARD is not None
             else R.score(texts, dirs_rep, actor, tok, device, a))
        torch.cuda.synchronize(); t_s = time.time() - t_s
        adv = R.compute_advantages(r, Bl, G, "group")
        L = p_len + max(len(g) for g in gen_ids)
        ids = torch.full((Bl * G, L), tok.pad_token_id, dtype=torch.long); attn = torch.zeros((Bl * G, L), dtype=torch.long)
        old_lp = torch.zeros((Bl * G, L - p_len)); known = torch.zeros((Bl * G, L - p_len), dtype=torch.bool)
        for i, g in enumerate(gen_ids):
            ids[i, :p_len] = prompt.cpu(); ids[i, p_len : p_len + len(g)] = torch.tensor(g); attn[i, : p_len + len(g)] = 1
            old_lp[i, : len(g)] = -2.5; known[i, : len(g)] = True
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t_u = time.time()
        st = update_disagg(actor, opt, submodule, ids, attn, p_len, marker, old_lp, known, adv, dirs_rep, a, device, mb)
        torch.cuda.synchronize(); t_u = time.time() - t_u
        row = {"rollouts_per_rank": Bl * G, "score_s": t_s, "update_s": t_u, "ref_s": st["t_ref"], "fwd_bwd_s": st["t_fb"], "sync_s": st["t_sync"],
               "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "len_mean": float(lens.mean()), "L": L}
        rows.append(row)
        _log(tag, f"{Bl * G} rollouts/rank (L={L}): score {t_s:.1f}s | update {t_u:.1f}s (ref {st['t_ref']:.1f} fb {st['t_fb']:.1f} sync {st['t_sync']:.1f}) "
                  f"| peak {row['peak_gb']:.0f} GB")
        out["rows"] = rows
        json.dump(out, open(f"{a.work_dir}/bench_trainer_r{rank}.json", "w"), indent=1)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
    _log(tag, "bench done")


# ==============================================================================================
# LAUNCHER
# ==============================================================================================
def _n_gpus():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return sum(1 for l in out.splitlines() if l.startswith("GPU "))
    except Exception:  # noqa
        return 0


def run_launch(a, argv):
    n = _n_gpus()
    X, Y = a.n_rollout, a.n_trainer
    bench = a.role == "bench"
    assert X + Y == n, f"n_rollout + n_trainer = {X + Y} but the container has {n} GPUs"
    work = a.work_dir
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    for d in ("lora", "queue"):
        os.makedirs(f"{work}/{d}", exist_ok=True)
    base_env = os.environ.copy()
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        base_env.pop(k, None)
    procs = []

    def spawn(role, gpu, rank, world):
        env = dict(base_env, CUDA_VISIBLE_DEVICES=str(gpu), DISAGG_RANK=str(rank), DISAGG_WORLD=str(world))
        if role in ("trainer", "bench-trainer") and base_env.get("DISAGG_TRAINER_PYTHONPATH"):
            # Hopper: fla 0.5.2 refuses its gated chunk_bwd_dqkwg on Triton 3.4-3.7.0 (fla #640, wrong results). The HF
            # trainer children (no vLLM) get a newer Triton from this dir; the vLLM rollout children keep torch's pin.
            env["PYTHONPATH"] = base_env["DISAGG_TRAINER_PYTHONPATH"] + os.pathsep + base_env.get("PYTHONPATH", "")
        child_argv = [x for x in argv]
        child_argv[child_argv.index("--role") + 1] = role
        tagp = {"trainer": "T", "rollout": "R", "bench-rollout": "B", "bench-trainer": "BT"}[role] + str(rank)
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + child_argv, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs.append((tagp, role, p))

        def pump():
            for line in p.stdout:
                sys.stdout.write(f"[{tagp}] {line}"); sys.stdout.flush()
        threading.Thread(target=pump, daemon=True).start()
        return p

    if bench:   # trainer bench on GPUs [0,Y) and rollout bench on [Y,N), concurrently, independent
        for k in range(Y):
            spawn("bench-trainer", k, k, Y)
        for r in range(X):
            spawn("bench-rollout", Y + r, r, X)
    else:
        for k in range(Y):
            spawn("trainer", k, k, Y)
        for r in range(X):
            spawn("rollout", Y + r, r, X)
    print(f"[launch] {len(procs)} processes on {n} GPUs: {[(t, role) for t, role, _ in procs]}", flush=True)
    rc = 0
    try:
        while True:
            alive = [(t, role, p) for t, role, p in procs if p.poll() is None]
            dead = [(t, role, p) for t, role, p in procs if p.poll() is not None]
            for t, role, p in dead:
                if p.returncode != 0 and (role != "rollout" or not _stop_requested(work)):
                    if bench:   # benches are independent: let the others finish, report the failure at the end
                        if getattr(p, "_reported", False) is False:
                            print(f"[launch] {t} ({role}) exited rc={p.returncode}", flush=True); p._reported = True
                        rc = rc or p.returncode
                    else:
                        print(f"[launch] {t} ({role}) exited rc={p.returncode} -> aborting", flush=True); rc = p.returncode
            if rc and not bench:
                break
            if bench:
                if not alive:
                    break
            else:
                if not [x for x in alive if x[1] == "trainer"]:
                    _atomic_write_text(f"{work}/STOP", "trainers exited")
                    t_end = time.time()
                    while any(p.poll() is None for _, _, p in procs) and time.time() - t_end < 60:
                        time.sleep(1)
                    break
            time.sleep(2)
    finally:
        for t, role, p in procs:
            if p.poll() is None:
                p.terminate()
        time.sleep(5)
        for t, role, p in procs:
            if p.poll() is None:
                p.kill()
    print(f"[launch] done rc={rc}", flush=True)
    return rc


def main():
    argv = sys.argv[1:]
    a = parse_args(argv)
    if a.role in ("launch", "bench"):
        sys.exit(run_launch(a, argv))
    if a.role == "trainer":
        run_trainer(a)
    elif a.role == "rollout":
        run_rollout(a)
    elif a.role == "bench-rollout":
        run_bench_rollout(a)
    elif a.role == "bench-trainer":
        run_bench_trainer(a)


if __name__ == "__main__":
    main()
