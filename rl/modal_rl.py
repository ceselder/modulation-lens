"""Modal app: Dr.GRPO RL (train/rl.py) on a single 8xB200 container.

The trainer lives at train/rl.py in this repo; it is mounted into the container at
/pmx/RL/rl_hf.py (its original path on the training box, which the commands below expect),
with mxf/ importable via PYTHONPATH=/pmx/helpers. Port of the B300-box run
`big_rl_longhz_dp4_lp025`. Data lives on the `maemm-data` Volume (uploaded via
`modal volume put`):
    /data/pool_rl_mix        direction bank (vecs.f32 750k x 5120 f32 + records.jsonl + build_stats.json)
    /data/sft_init           init LoRA adapter (also the frozen KL reference)
    /data/sae/{ae.pt,maxacts.pt}, /data/pool_heldout, /data/eval_universal_ho   (future evals)
    /data/hf_cache           HF_HOME (Qwen/Qwen3.6-27B downloads once, persists)
    /data/ckpts              output checkpoints (step_25, step_50, ... final)

Needs two Modal secrets in your workspace: `maemm-hf` (HF_TOKEN) and `maemm-wandb`
(WANDB_API_KEY). Launch from the repo root:
    modal run --detach modal_rl.py
Options:
    --backend gloo           if NCCL hangs at the first collective (box-specific bug; try nccl first)
    --total-steps N          override step count (default 400)
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent   # repo root (this launcher lives one level down)

app = modal.App("maemm-rl-8xb200")

# torch 2.10.0+cu128 == the box venv; cu128 wheels carry sm_100 (B200) kernels.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
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
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

TRAIN_ARGS = [
    "--bank-file", "vecs.f32",
    # Start RL FRESH from the SFT adapter (= start policy AND KL ref). Warm-starting from the
    # deep-RL dp4/step_100 collapsed even at LP 1.0 (fresh optimizer + weak KL anchor + policy
    # already near the reward-hack cliff). SFT-init is the proven-stable pattern (uni_rl: 270
    # steps no collapse; original longhz: 100 steps stable).
    "--init-adapter", "/data/sft_init",
    "--lr", "1e-5",
    "--reward-metric", "cosine",
    "--reward-scale", "1000",
    "--min-new-tokens", "16",
    "--max-new-tokens", "96",
    "--len-penalty-start", "16",
    # PAPER RUN (final config): LP 0.25/tok, from-SFT. Earlier 0.25 collapses were warm-start
    # runs (deep-RL init + fresh optimizer), not the LP value itself.
    "--len-penalty-per-tok", "0.25",
    # --div-coef is a launch parameter (see train()/main); paper run uses 1000 (gate-masked).
    "--kl-coef", "0.005",
    "--groups-per-step", "128",  # 128 % 8 == 0 -> 16 whole groups per rank = 2048 rollouts/step
    "--group-size", "16",
    "--rollout-chunk", "64",
    # micro-batch 4 (box used 8): update() peaked OOM on 178GB B200s at gen len ~42. Pure grad-
    # accumulation slicing — global-token-normalized loss makes gradients identical to mb=8.
    "--micro-batch", "4",
    "--score-batch", "24",
    "--save-every", "25",
    "--eval-every", "0",
    "--sae-eval-every", "0",
    "--save-dir", "/data/ckpts_v2",
    "--run-name", "maemm-8xb200-paper-v3",   # v3 = resume-from-step_250 era (kl 0.1 leash)
]


@app.function(
    image=image,
    gpu="B200:8",
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=86400,
)
def train(backend: str = "gloo", total_steps: int = 400, div_coef: float = 0.0,
          resume_from: str = "", step_offset: int = 0, wandb_id: str = "",
          save_dir: str = "", run_name: str = "", groups_per_step: int = 0, save_every: int = 0):
    # resume_from: path to a step_N ckpt dir -> becomes --init-adapter, with --ref-adapter kept at
    # the SFT init (KL anchor NEVER re-anchors to the resume ckpt) + --step-offset for global-step
    # continuity + optional --wandb-id to continue the same wandb run. The trainer auto-loads
    # <resume_from>/optim.pt (AdamW moments) when present.
    # save_dir/run_name/groups_per_step/save_every: TRAIN_ARGS overrides for throwaway smoke runs.
    # backend MUST be gloo: rl_hf.py's _ddp_sync_grads/all_gather run on CPU tensors by design
    # ("gloo: CPU tensor, no NCCL anywhere") — under NCCL they raise
    # "RuntimeError: No backend type associated with device type cpu" at the first collective.
    import os
    import shutil
    import subprocess
    import threading
    import time

    # ---- collapse-fix guard: the mounted trainer MUST carry the gate-masked diversity bonus
    # (_gmask). Refuse to train on unpatched code — the un-masked div bonus caused the box
    # collapse (reward 390->53, gate 90%->8%). ----
    with open("/pmx/RL/rl_hf.py") as f:
        _src_lines = f.read().splitlines()
    _hits = [f"{i + 1}: {l.strip()}" for i, l in enumerate(_src_lines) if "_gmask" in l]
    assert len(_hits) >= 2, f"collapse fix (_gmask) MISSING from mounted rl_hf.py — got {_hits}"
    _lp_hits = [f"{i + 1}: {l.strip()}" for i, l in enumerate(_src_lines)
                if "len_penalty_per_tok * over * " in l]
    assert _lp_hits, "gate-masked LEN PENALTY missing from mounted rl_hf.py (expected 'r = r - "\
                     "a.len_penalty_per_tok * over * (gate.float() ...)')"
    print("[modal] collapse-fix check OK (_gmask div + gate-masked len-penalty):\n  "
          + "\n  ".join(_hits + _lp_hits), flush=True)

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
    local_pool = "/root/pool_rl_mix"
    if not os.path.exists(local_pool):
        shutil.copytree("/data/pool_rl_mix", local_pool)
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
    env["PYTHONPATH"] = "/pmx/helpers"
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
        args += ["--ref-adapter", "/data/sft_init", "--step-offset", str(step_offset)]
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
    if resume_from:
        # Resume-time RE-PARAMETERIZATION to raw-cosine reward units: every scale-coupled term
        # ÷1000, so reward/mean logs as ~0.42 (cosine) instead of ~420. Pure re-param — advantages,
        # clipped gradients and AdamW steps are unchanged (AdamW is scale-invariant modulo eps;
        # moments are fresh at the first wall-resume and saved/loaded at the NEW scale on later
        # legs, so the invariance holds exactly). Values are ABSOLUTE -> idempotent across resume
        # legs. Appended LAST so argparse last-wins overrides both TRAIN_ARGS and the launch
        # --div-coef. EXPECTED at the resume boundary: reward/mean + grad_norm wandb curves drop
        # 1000x. NOT scaled (scale-free): clip_eps 0.2, kl_cap 10 (nats), fluency/distinct/tri/
        # comp floors, lr (lr change is a separate step-400 decision).
        reparam = [
            ("--reward-scale", "1"),               # was 1000
            ("--len-penalty-per-tok", "0.00025"),  # was 0.25
            ("--div-coef", "1"),                   # was 1000
            ("--gate-penalty", "0.025"),           # was 25
            # kl 0.1 at cosine scale = a REAL leash to SFT (user-approved after the step-330+ gate
            # collapse: the scale-preserving 5e-6 was inert — kl_to_init drifted to 0.7 nats/tok
            # and the policy reward-hacked). NOT a pure re-param: this is the fix.
            ("--kl-coef", "0.1"),
            ("--max-grad-norm", "0.001"),          # was 1
        ]
        for _flag, _val in reparam:
            args += [_flag, _val]
        print(f"[modal] RESUME RE-PARAM to raw-cosine units: {reparam}", flush=True)

    cmd = [
        "torchrun", "--nproc_per_node=8", "--master_port=29531", "RL/rl_hf.py",
        "--data-dir", local_pool,
        "--total-steps", str(total_steps),
        "--div-coef", str(div_coef),
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
    download) and runs 2 tiny world=1 steps (no wandb, ckpts to /tmp)."""
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

    local_pool = "/root/pool_rl_mix"
    if not os.path.exists(local_pool):
        shutil.copytree("/data/pool_rl_mix", local_pool)
    print("[modal-smoke] pool staged", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"
    os.makedirs("/tmp/wandb", exist_ok=True)
    cmd = [
        "python", "RL/rl_hf.py",
        "--data-dir", local_pool,
        "--bank-file", "vecs.f32",
        "--init-adapter", "/data/warmstart",
        "--lr", "1e-5", "--reward-metric", "cosine", "--reward-scale", "1000",
        "--min-new-tokens", "16", "--max-new-tokens", "96",
        "--len-penalty-start", "16", "--len-penalty-per-tok", "0.25",
        "--div-coef", "0", "--kl-coef", "0.03",
        "--groups-per-step", "2", "--group-size", "4",
        "--rollout-chunk", "8", "--micro-batch", "4", "--score-batch", "8",
        "--total-steps", "2", "--save-every", "0",
        "--eval-every", "0", "--sae-eval-every", "0",
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


# ---- paper-run auto-resume supervisor: Modal caps functions at 24h, but the 1000-step paper run
# needs ~50h (2048 rollouts/step @ ~180s). This scheduled function respawns `train` in RESUME mode
# (latest ckpt + optim.pt + --ref-adapter /data/sft_init + --step-offset + same wandb run)
# whenever the trainer is dead and the newest /data/ckpts_v2 step is < PAPER_TOTAL_STEPS. ----
PAPER_TOTAL_STEPS = 1000
PAPER_DIV_COEF = 1000.0
PAPER_CKPT_DIR = "/data/ckpts_v2"
PAPER_WANDB_ID = "uxglv0vr"          # maemm-8xb200-paper-v3 (kl-0.1 leash era) — wall-resumes
                                     # continue this run; buoi7l3k = the collapsed v2 series
RESUME_STATE = "/data/resume_state.json"
SPAWN_COOLDOWN_S = 2 * 3600          # never double-spawn while a fresh leg warms up
STALE_CKPT_S = 110 * 60              # save-every 25 steps ≈ 75 min; older = trainer presumed dead


@app.function(schedule=modal.Period(minutes=20), volumes={"/data": vol}, timeout=600)
def supervisor():
    import glob
    import json
    import os
    import time

    vol.reload()
    if os.path.exists("/data/resume_paused"):
        # HOLD: resuming with the current (inert, 5e-6 reparam) kl_coef would re-collapse the same
        # way — the inert KL leash is the diagnosed root cause of the step-330+ gate collapse.
        # The go comes with a REAL kl value (~0.1 at cosine scale); until then: NO auto-spawn.
        print("[supervisor] auto-resume PAUSED (/data/resume_paused present) — awaiting explicit "
              "go with a finalized KL leash; no spawn will happen", flush=True)
        return
    steps = sorted(int(p.rsplit("_", 1)[-1]) for p in glob.glob(f"{PAPER_CKPT_DIR}/step_*")
                   if p.rsplit("_", 1)[-1].isdigit())
    latest = steps[-1] if steps else 0
    if os.path.exists(f"{PAPER_CKPT_DIR}/final") or latest >= PAPER_TOTAL_STEPS:
        print(f"[supervisor] paper run COMPLETE (latest step_{latest}) — nothing to do", flush=True)
        return
    if not steps:
        print("[supervisor] no ckpts yet — first leg is launched manually, not spawning", flush=True)
        return
    # PRIMARY aliveness signal: ckpt freshness (version-agnostic — a live trainer saves every
    # save_every*step_s ≈ 76 min). get_current_stats is scoped to the LATEST function version, so a
    # live container from an older deploy reads as 0 runners: stats may only CONFIRM life, never death.
    newest_m = max((os.path.getmtime(p) for p in
                    glob.glob(f"{PAPER_CKPT_DIR}/step_*/adapter_model.safetensors")), default=0.0)
    age_min = (time.time() - newest_m) / 60
    if age_min * 60 < STALE_CKPT_S:
        print(f"[supervisor] trainer alive (newest ckpt step_{latest}, {age_min:.0f} min old)", flush=True)
        return
    try:
        runners = modal.Function.from_name("maemm-rl-8xb200", "train").get_current_stats().num_total_runners
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
    train_fn = modal.Function.from_name("maemm-rl-8xb200", "train")
    resume_step = latest
    if os.path.exists("/data/resume_override.json"):
        # human-pinned resume point (e.g. latest ckpts are from a degraded/collapsing policy)
        ov = json.load(open("/data/resume_override.json"))
        if ov.get("step") in steps:
            resume_step = ov["step"]
            print(f"[supervisor] resume OVERRIDE active: step_{resume_step} "
                  f"(reason: {ov.get('reason', 'n/a')}) instead of latest step_{latest}", flush=True)
    ck = f"{PAPER_CKPT_DIR}/step_{resume_step}"
    offset = resume_step + 1  # step_N is saved after iteration N completes
    print(f"[supervisor] trainer DEAD at step_{latest} < {PAPER_TOTAL_STEPS} — resuming: "
          f"init={ck} ref=/data/sft_init offset={offset} wandb={PAPER_WANDB_ID}", flush=True)
    call = train_fn.spawn(backend="gloo", total_steps=PAPER_TOTAL_STEPS, div_coef=PAPER_DIV_COEF,
                          resume_from=ck, step_offset=offset, wandb_id=PAPER_WANDB_ID)
    json.dump({"last_spawn_ts": time.time(), "call": call.object_id, "from_step": latest},
              open(RESUME_STATE, "w"))
    vol.commit()
    print(f"[supervisor] resume leg spawned: {call.object_id}", flush=True)


@app.local_entrypoint()
def main(backend: str = "gloo", total_steps: int = 400, div_coef: float = 0.0):
    train.remote(backend=backend, total_steps=total_steps, div_coef=div_coef)


@app.local_entrypoint()
def run_prewarm():
    prewarm.remote()


@app.local_entrypoint()
def run_smoke():
    smoke.remote()
