"""Modal app: Dr.GRPO RL, LAST-5 run — peak-in-last-5 reward on the 4-family direction pool.

Copy of modal_rl.py (the paper app) with the last-5 config baked in:
    data      /data/pool_rl_last5   (100k dirs: 25k realact + 25k cluster + 25k realact_long
                                     + 25k unit SAE ENCODER columns unit(W_enc[:,f]) — the
                                     mxf.sae.enc_dirs convention; built by modal_pool_last5.py)
    init/ref  /data/sft_mix/last5_rp/final   (the last5_rp SFT-final adapter = start policy
                                              AND frozen KL reference; SFT-init is the
                                              proven-stable pattern — never warm-start deep-RL)
    reward    --reward-window-last 5 --reward-topk 1  (anti-smear: max cos over only the LAST
                                     5 kept tokens — the snippet-locality fix)
    ckpts     /data/ckpts_last5, wandb run last5_rp_rl
    units     RAW-COSINE from step 0: reward-scale 1, LP 2.5e-4/tok, gate-penalty 0.025,
              max-grad-norm 1e-3, div-coef 1, KL 0.1 (the kl-0.1 leash that fixed the paper
              run's step-330 gate collapse — modal_rl.py only reaches these values via its
              resume-time re-param; a FRESH run starts there directly so every leg, first or
              resumed, runs the same proven-final config). To revert to the paper first-leg
              x1000 units: reward-scale 1000, LP 0.25, gate-penalty 25, max-grad-norm 1,
              kl 0.005, div-coef 1000.

Launch (MODAL_PROFILE=safety-sahan, AFTER /data/sft_mix/last5_rp/final exists):
    modal deploy modal_rl_last5.py                    # registers train + auto-resume supervisor
    modal run --detach modal_rl_last5.py::main        # first leg (gloo, 400 steps, div 1)
After the first leg starts, write its wandb run id to /data/ckpts_last5/wandb_id.txt so
wall-resume legs continue the same wandb run (else each leg logs to a fresh run).
Options:
    --backend gloo    REQUIRED default: rl.py's grad all-reduce + reward all_gather run on CPU
                      tensors by design; under NCCL they raise "No backend type associated with
                      device type cpu" at the first collective (see TRAINING_LEDGER.md).
    --total-steps N   override step count (default 400; keep RL_TOTAL_STEPS in sync for resume)

Needs Modal secrets `maemm-hf` (HF_TOKEN) and `maemm-wandb` (WANDB_API_KEY).
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent   # repo root (this launcher lives one level down)

import os  # noqa: E402  (N_GPU below reads the env at import time)

app = modal.App("maemm-rl-last5-v15")
# GPUs per arm: Modal has only handed us ~8 B200 at a time today; two 8-GPU arms never scheduled together.
# RL_NGPU=4 at deploy time -> each arm on 4xB200 (groups/rank doubles, ~2x step time, both arms run in parallel).
N_GPU = int(os.environ.get("RL_NGPU", "8"))   # deploy-time value -> the @app.function gpu= request below


def _visible_gpus():
    """GPUs actually attached to THIS container (N_GPU is only trustworthy at deploy time: the module is
    re-imported inside the container without RL_NGPU, so torchrun's nproc must come from the hardware)."""
    import subprocess
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        n = sum(1 for l in out.splitlines() if l.startswith("GPU "))
        return n or N_GPU
    except Exception:  # noqa
        return N_GPU

# torch 2.10.0+cu128 == the box venv; cu128 wheels carry sm_100 (B200) kernels.
image = (
    modal.Image.debian_slim(python_version="3.12")  # vllm-lens needs >=3.12
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # vLLM rollout engine (pins torch==2.10.0 -> matches the layer above) + vllm_lens steering plugin
    # (1.1.0 = the version proven on this model in scripts/vllm_smoke.py). transformers is re-pinned below.
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
    .pip_install(
        "transformers==5.15.0",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "wandb==0.28.2",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    .add_local_file(REPO / "rl" / "rl.py", "/pmx/RL/rl_hf.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "eval" / "eval_universal.py", "/pmx/eval/eval_universal.py")   # inline eval scoring
    .add_local_file(REPO / "eval" / "inline_extra_evals.py", "/pmx/RL/inline_extra_evals.py")   # autointerp/locality/WildChat/adversarial inline
    .add_local_file(REPO / "eval" / "snippet_locality.py", "/pmx/eval/snippet_locality.py")
    .add_local_file(REPO / "eval" / "autointerp_detection.py", "/pmx/eval/autointerp_detection.py")
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

POOL_DIR = "/data/banks/everything"              # v15 (user, Sep 2): "train on LITERALLY EVERYTHING" — 5 x 100k even mix (short-ctx acts, long-ctx acts, SAE enc features, BSF subspace projections, probes), eval ids excluded + leakage-asserted (modal_bank_everything.py). Was /data/pool_rl_last5 (realact+probes; contains 128/512 eval cluster dirs!)
SFT_INIT = "/data/sft_mix/last5_rp/final"        # SFT-final adapter (init AND frozen KL ref)
CKPT_DIR = "/data/ckpts_last5_v15_g8"  # v15 = 8 rollouts/prompt x 128 prompts (user: "8x128"), inline eval, grad-ckpt, EasyNLA-matched recipe (per-group std adv, kl .01, raw cosine, hinged len pen, cap-hit -2, seq-mean), lr 1e-5

TRAIN_ARGS = [
    "--bank-file", "vecs.f32",
    # Start RL FRESH from the SFT adapter (= start policy AND KL ref). Warm-starting from a
    # deep-RL ckpt collapsed even at LP 1.0 (fresh optimizer + weak KL anchor + policy already
    # near the reward-hack cliff). SFT-init is the proven-stable pattern.
    "--init-adapter", SFT_INIT,
    "--lr", "1e-5",                              # user (Sep 2): keep 1e-5 (EasyNLA runs 1e-4; not adopted)
    "--reward-metric", "cosine",
    # ---- RAW-COSINE units from step 0 (paper run's post-resume re-param values, ABSOLUTE).
    # The kl 0.1 leash is the fix for the paper run's step-330+ gate collapse (the x1000-era
    # kl 0.005 ≈ 5e-6 at cosine scale was inert). Baking them here (instead of modal_rl.py's
    # resume-time re-param) means the first leg never runs the collapse-prone inert-KL config. ----
    "--reward-scale", "1",                       # v15: raw cosine (EasyNLA uses raw -MSE; per-group std-norm makes the scale irrelevant except for the shaping terms below)
    "--len-penalty-start", "32",                # v15: EasyNLA hinge = max_new_tokens - 64 (their 256-64=192) -> 96-64 = 32
    "--len-penalty-per-tok", "0.01",            # v15: EasyNLA 0.01/tok past the hinge, in raw reward units
    "--no-gates",                                # v11: NO fluency/distinct gate (user call; the collapse analysis showed the gate is not the stabilizer — KL is). len-penalty stays.
    "--kl-coef", "0.01",                         # v15: EasyNLA k3 kl_beta default
    "--adv-mode", "group",                      # v15: EasyNLA = standard GRPO (r - group_mean) / (group_std + 1e-6), no batch norm
    "--adam-eps", "1e-8",                        # v15: EasyNLA AdamW8bit defaults
    "--adam-betas", "0.9", "0.95",               # v15: EasyNLA betas
    "--loss-agg", "seq",                         # v15: EasyNLA per-rollout mean then mean over rollouts (not token-mean)
    "--trunc-reward", "-2",                      # v15: EasyNLA cap-hit failure reward (kept in the group, still trains)
    "--max-grad-norm", "1",                      # paper (clips the ~x1000 grads; Adam handles it)
    # ---- the LAST-5 reward: max per-token cosine over only the last 5 kept content tokens
    # (anti-smear — the snippet-locality eval showed max-over-ALL lets the policy smear the
    # feature across every token). topk 1 = plain max within the window. ----
    "--reward-window-last", "5",
    "--reward-topk", "1",
    "--min-new-tokens", "8",     # v14 (user): min generation 8 tokens (len penalty also from 8)
    "--max-new-tokens", "96",
    "--groups-per-step", "128", # user (Sep 2): 128 prompts/step (16/rank) x 16 = 2048 rollouts/step
    "--group-size", "8",         # user (Sep 2): 8 rollouts per prompt ("8x128"; 16x128 killed at step 5 to make room)
    "--rollout-chunk", "64",
    "--logp-chunk", "16",        # old_logp recompute chunk (fp32 248k-vocab logits: 64 seqs ~13 GB peak OOM'd next to the vLLM engine)
    "--rollout-engine", "vllm",  # v11: per-rank vLLM engine + vllm_lens steering; HF only recomputes old_logp
    "--vllm-gpu-mem", "0.33",   # v15d: 59 GB engine (was 64) -> headroom for the inline-eval SAE (2.7 GB, decoder dropped) next to the 99 GB update peak
    # micro-batch 4 (box used 8): update() peaked OOM on 178GB B200s at gen len ~42. Pure grad-
    # accumulation slicing — global-token-normalized loss makes gradients identical to mb=8.
    "--micro-batch", "3",   # v15d: mb 4 + grad-ckpt OOM'd in loss.backward() (112 GB HF + 64 GB vLLM); checkpointing did NOT lower the peak on this model -> back to the measured-safe mb 3 (99 GB peak), no grad-ckpt
    "--inline-eval-every", "10",   # v15b: held-out eval suite INSIDE the trainer on all 8 GPUs every save (no separate runner)
    "--eval-n-per-family", "128",  # v15e: 512/family (=daemon protocol) cost 1474 s at step 1 on 4 ranks (~25 min/ckpt); 128 -> ~6 min (noisier per-ckpt means, same subset every ckpt)
    "--ref-micro-batch", "16",   # KL ref logps in one no-grad pass (2 adapter switches/step instead of 2/micro-batch)
    "--score-batch", "64",  # gates off -> score() is a read_resid early-exit pass; bigger batch = fewer forwards
    "--save-every", "10",   # legs die at ~step 22 (B200 eviction on shared ws); 25 never saved -> resume-chain never bootstrapped. 10 => step_10/step_20 land within a leg.
    "--save-dir", CKPT_DIR,
    "--run-name", "rl_everything_8x128_last5win",
]


@app.function(
    image=image,
    gpu=f"B200:{N_GPU}",
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
        modal.Secret.from_name("maemm-openrouter"),
        modal.Secret.from_name("maemm-anthropic"),    # native Sonnet 5 judge (key + anthropic-workspace-id); preferred over OpenRouter by inline_extra_evals   # judge key for the inline extra evals (Anthropic keys blocked: workspace id)
    ],
    timeout=86400,
)
def train(backend: str = "gloo", total_steps: int = 400,
          resume_from: str = "", step_offset: int = 0, wandb_id: str = "",
          save_dir: str = "", run_name: str = "", groups_per_step: int = 0, save_every: int = 0,
          extra_args: str = ""):
    # extra_args: whitespace-split, appended LAST (argparse last-wins) — per-arm overrides for sweeps, e.g. "--kl-coef 0.005".
    # resume_from: path to a step_N ckpt dir -> becomes --init-adapter, with --ref-adapter kept at
    # the SFT init (KL anchor NEVER re-anchors to the resume ckpt) + --step-offset for global-step
    # continuity + optional --wandb-id to continue the same wandb run. The trainer auto-loads
    # <resume_from>/optim.pt (AdamW moments) when present. NO resume-time re-param here (unlike
    # modal_rl.py): TRAIN_ARGS already carries the raw-cosine values, identical on every leg.
    # save_dir/run_name/groups_per_step/save_every: TRAIN_ARGS overrides for throwaway smoke runs.
    # backend MUST be gloo: rl_hf.py's _ddp_sync_grads/all_gather run on CPU tensors by design
    # ("gloo: CPU tensor, no NCCL anywhere") — under NCCL they raise
    # "RuntimeError: No backend type associated with device type cpu" at the first collective.
    import os
    import shutil
    import subprocess
    import threading
    import time

    # ---- trainer guards: the mounted rl_hf.py must be the SIMPLIFIED trainer (diversity /
    # first-token shaping REMOVED, in-trainer evals removed) with the sink-prepended clean-base reward
    # and the last-N reward window. Fail fast on a stale mount. ----
    with open("/pmx/RL/rl_hf.py") as f:
        _src = f.read()
    assert "def compute_advantages" in _src, "simplified trainer missing (no compute_advantages)"
    assert "meanact" not in _src and "firsttok" not in _src.replace("--firsttok-coef", "").replace("a.firsttok_coef", ""), \
        "stale trainer: diversity/first-token shaping still present"
    assert "sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id" in _src, \
        "sink-prepended clean-base reward missing"
    assert "reward_window_last" in _src, "last-N reward window missing"
    print("[modal] guards OK (simplified trainer: sink reward + last-N window, no div/firsttok)", flush=True)

    # ---- input guards: fail FAST and CLEARLY if the pool or the SFT-final adapter is missing
    # (the SFT run writes `final` only when it completes — do not launch before that). ----
    for p in (f"{POOL_DIR}/vecs.f32", f"{POOL_DIR}/records.jsonl", f"{POOL_DIR}/build_stats.json"):
        assert os.path.exists(p), f"direction pool incomplete: missing {p} (run modal_pool_last5.py)"
    init = resume_from or SFT_INIT
    assert os.path.exists(f"{init}/adapter_model.safetensors") and \
        os.path.exists(f"{init}/adapter_config.json"), (
        f"init adapter incomplete at {init} — if this is {SFT_INIT}, the last5_rp SFT run has "
        "not written `final` yet; wait for it (do NOT init from a step_N ckpt)")

    os.environ["HF_HOME"] = "/data/hf_cache"

    # single-flight base-model download into the persistent volume (avoids 8 ranks racing)
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download("Qwen/Qwen3.6-27B")
    vol.commit()
    print(f"[modal] base model in cache ({time.time() - t0:.0f}s)", flush=True)

    # stage the direction bank onto container-local disk: memmap over the volume FUSE mount is
    # the one thing we don't trust, and per-step random row reads are faster off local NVMe.
    t0 = time.time()
    local_pool = "/root/pool_rl_last5"
    if not os.path.exists(local_pool):
        shutil.copytree(POOL_DIR, local_pool)
    print(f"[modal] pool staged to {local_pool} ({time.time() - t0:.0f}s)", flush=True)

    # periodic volume commit so checkpoints land even if the container dies mid-run
    def _committer():
        while True:
            time.sleep(300)
            try:
                vol.commit()
            except Exception as e:
                print(f"[modal] vol.commit failed: {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers:/pmx/eval"
    env["DDP_BACKEND"] = backend
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"          # /pmx is a read-only mount; wandb writes to cwd otherwise
    # ranks must load PURELY from the validated cache: 8 concurrent hub re-resolutions returned
    # spurious "does not appear to have a file named model-0000X-of-00015.safetensors" even though
    # the hub file exists and the cached snapshot (same sha) is complete. snapshot_download above
    # (driver, online) already guarantees completeness.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # B200 = 178GB (box B300 = 288GB): variable-length RL batches fragment the caching allocator —
    # rank 0 OOM'd at step 3 with 159GB allocated failing a 24MB alloc. expandable_segments is the
    # canonical fix (recommended by the OOM message itself).
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs("/tmp/wandb", exist_ok=True)

    args = list(TRAIN_ARGS)

    def _set(flag, val):
        args[args.index(flag) + 1] = str(val)

    if resume_from:
        _set("--init-adapter", resume_from)
        args += ["--ref-adapter", SFT_INIT, "--step-offset", str(step_offset)]
        if wandb_id:
            args += ["--wandb-id", wandb_id]
    if save_dir:
        _set("--save-dir", save_dir)
    if run_name:
        _set("--run-name", run_name)
    if groups_per_step:
        _set("--groups-per-step", groups_per_step)
    if save_every:
        _set("--save-every", save_every)
    if extra_args:
        args += extra_args.split()

    cmd = [
        "torchrun", f"--nproc_per_node={_visible_gpus()}", "--master_port=29531", "RL/rl_hf.py",
        "--data-dir", local_pool,
        "--total-steps", str(total_steps),
    ] + args
    print("[modal] launching:", " ".join(cmd), f"(DDP_BACKEND={backend})", flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"torchrun exited rc={rc} (backend={backend})")
    print("[modal] training complete", flush=True)


@app.function(
    image=image,
    gpu="B200:1",
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=7200,
)
def smoke():
    """1xB200 pipeline validation: pre-warms /data/hf_cache (so the 8x run skips the 55GB
    download) and runs 2 tiny world=1 steps of the EXACT last-5 config (no wandb, ckpts to /tmp).
    Needs the SFT-final adapter — run only after /data/sft_mix/last5_rp/final exists."""
    import os
    import shutil
    import subprocess
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download("Qwen/Qwen3.6-27B")
    vol.commit()
    print(f"[modal-smoke] base model cached ({time.time() - t0:.0f}s)", flush=True)

    local_pool = "/root/pool_rl_last5"
    if not os.path.exists(local_pool):
        shutil.copytree(POOL_DIR, local_pool)
    print("[modal-smoke] pool staged", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers:/pmx/eval"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"
    os.makedirs("/tmp/wandb", exist_ok=True)
    cmd = [
        "python", "RL/rl_hf.py",
        "--data-dir", local_pool,
        "--bank-file", "vecs.f32",
        "--init-adapter", SFT_INIT,
        "--lr", "1e-5", "--reward-metric", "cosine", "--reward-scale", "1",
        "--min-new-tokens", "16", "--max-new-tokens", "96",
        "--len-penalty-start", "8", "--len-penalty-per-tok", "0.00025",
        "--gate-penalty", "0.025", "--max-grad-norm", "0.001",
        "--reward-window-last", "5", "--reward-topk", "1",
        "--kl-coef", "0.1",
        "--groups-per-step", "2", "--group-size", "4",
        "--rollout-chunk", "8", "--micro-batch", "4", "--score-batch", "8",
        "--rollout-engine", "vllm", "--vllm-gpu-mem", "0.36",
        "--total-steps", "2", "--save-every", "0",
        "--save-dir", "/tmp/smoke_ckpt", "--no-wandb",
    ]
    print("[modal-smoke] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"smoke exited rc={rc}")
    print("[modal-smoke] OK", flush=True)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=7200,
    cpu=8,
)
def prewarm():
    """CPU-only: download the base model into the persistent HF cache on the volume."""
    import os
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download("Qwen/Qwen3.6-27B")
    vol.commit()
    print(f"[modal-prewarm] base model cached ({time.time() - t0:.0f}s)", flush=True)


# ---- auto-resume supervisor: Modal caps functions at 24h; this scheduled function respawns
# `train` in RESUME mode (latest ckpt + optim.pt + --ref-adapter SFT_INIT + --step-offset + the
# wandb id from /data/ckpts_last5/wandb_id.txt when present) whenever the trainer is dead and the
# newest CKPT_DIR step is < RL_TOTAL_STEPS. Pointed EXCLUSIVELY at the last5 run — its state/pause
# files are distinct from the paper run's (/data/resume_state.json, /data/resume_paused). ----
RL_TOTAL_STEPS = 400                       # keep in sync with the launch --total-steps
RL_DIV_COEF = 1.0                          # paper 1000 at x1000 scale -> 1 at raw-cosine scale
WANDB_ID_FILE = f"{CKPT_DIR}/wandb_id.txt"  # write the first leg's wandb run id here
RESUME_STATE = "/data/resume_state_last5.json"
RESUME_PAUSED = "/data/resume_paused_last5"
RESUME_OVERRIDE = "/data/resume_override_last5.json"
SPAWN_COOLDOWN_S = 60 * 60           # v6: ~280s/step, save-every 10 -> first ckpt ~52min. cooldown must exceed that so a warming leg isn't double-spawned.
STALE_CKPT_S = 90 * 60               # v6: save-every 10 @ ~280s/step ≈ 47min cadence; >90min w/ no live runner = dead (eviction). MUST be > save-cadence (v5's 45min was too tight -> false-positive resume spawn).


@app.function(schedule=modal.Period(minutes=20), volumes={"/data": vol}, timeout=600)
def supervisor():
    import glob
    import json
    import os
    import time

    vol.reload()
    if os.path.exists(RESUME_PAUSED):
        print(f"[supervisor] auto-resume PAUSED ({RESUME_PAUSED} present) — no spawn will happen",
              flush=True)
        return
    steps = sorted(int(p.rsplit("_", 1)[-1]) for p in glob.glob(f"{CKPT_DIR}/step_*")
                   if p.rsplit("_", 1)[-1].isdigit())
    latest = steps[-1] if steps else 0
    if os.path.exists(f"{CKPT_DIR}/final") or latest >= RL_TOTAL_STEPS:
        print(f"[supervisor] last5 run COMPLETE (latest step_{latest}) — nothing to do", flush=True)
        return
    if not steps:
        print("[supervisor] no ckpts yet — first leg is launched manually, not spawning", flush=True)
        return
    # PRIMARY aliveness signal: ckpt freshness (version-agnostic — a live trainer saves every
    # save_every*step_s ≈ 76 min). get_current_stats is scoped to the LATEST function version, so a
    # live container from an older deploy reads as 0 runners: stats may only CONFIRM life, never death.
    newest_m = max((os.path.getmtime(p) for p in
                    glob.glob(f"{CKPT_DIR}/step_*/adapter_model.safetensors")), default=0.0)
    age_min = (time.time() - newest_m) / 60
    if age_min * 60 < STALE_CKPT_S:
        print(f"[supervisor] trainer alive (newest ckpt step_{latest}, {age_min:.0f} min old)", flush=True)
        return
    try:
        runners = modal.Function.from_name("maemm-rl-last5-v15", "train").get_current_stats().num_total_runners
        if runners > 0:
            print(f"[supervisor] ckpts stale ({age_min:.0f} min) but {runners} runner(s) live — waiting", flush=True)
            return
    except Exception as e:
        print(f"[supervisor] get_current_stats unavailable ({e})", flush=True)
    st = {}
    if os.path.exists(RESUME_STATE):
        st = json.load(open(RESUME_STATE))
    if time.time() - st.get("last_spawn_ts", 0) < SPAWN_COOLDOWN_S:
        print(f"[supervisor] in post-spawn cooldown (last spawn {st.get('call')}) — waiting", flush=True)
        return
    train_fn = modal.Function.from_name("maemm-rl-last5-v15", "train")
    resume_step = latest
    if os.path.exists(RESUME_OVERRIDE):
        # human-pinned resume point (e.g. latest ckpts are from a degraded/collapsing policy)
        ov = json.load(open(RESUME_OVERRIDE))
        if ov.get("step") in steps:
            resume_step = ov["step"]
            print(f"[supervisor] resume OVERRIDE active: step_{resume_step} "
                  f"(reason: {ov.get('reason', 'n/a')}) instead of latest step_{latest}", flush=True)
    ck = f"{CKPT_DIR}/step_{resume_step}"
    offset = resume_step + 1  # step_N is saved after iteration N completes
    wid = ""
    if os.path.exists(WANDB_ID_FILE):
        wid = open(WANDB_ID_FILE).read().strip()
    print(f"[supervisor] trainer DEAD at step_{latest} < {RL_TOTAL_STEPS} — resuming: "
          f"init={ck} ref={SFT_INIT} offset={offset} wandb={wid or '(new run)'}", flush=True)
    call = train_fn.spawn(backend="gloo", total_steps=RL_TOTAL_STEPS, div_coef=RL_DIV_COEF,
                          resume_from=ck, step_offset=offset, wandb_id=wid)
    json.dump({"last_spawn_ts": time.time(), "call": call.object_id, "from_step": latest},
              open(RESUME_STATE, "w"))
    vol.commit()
    print(f"[supervisor] resume leg spawned: {call.object_id}", flush=True)


@app.local_entrypoint()
def main(backend: str = "gloo", total_steps: int = 400, div_coef: float = 1.0):
    train.remote(backend=backend, total_steps=total_steps, div_coef=div_coef)


@app.local_entrypoint()
def run_prewarm():
    prewarm.remote()


@app.local_entrypoint()
def run_smoke():
    smoke.remote()
