"""Modal app: DISAGGREGATED Dr.GRPO/GRPO RL for the MAEMM inverter (rl/rl_disagg.py) -- X vLLM
rollout GPUs + Y HF trainer GPUs in ONE container. Dev iteration app (user: "a separate 4-GPU thing to
iterate on"); production is the same function on 8 GPUs with the chosen (X, Y).

Same image as modal_rl_last5_v15.py (torch 2.10 cu128, vllm 0.19 + vllm-lens 1.1.0, transformers 5.15,
peft 0.20) PLUS flash-linear-attention (fla): transformers' Qwen3.5 GatedDeltaNet picks fla's Triton
chunk kernel over its torch fallback when `fla` is importable (use_kernel_func_from_hub_with_fallback),
which is what makes the trainer's no-vLLM, no-grad-ckpt, micro-batch 16-32 update fit and run fast.
Mounts: rl/rl.py -> /pmx/RL/rl_hf.py (imported as a module, NEVER edited), rl/rl_disagg.py ->
/pmx/RL/rl_disagg.py, rl/fast_lens_ext.py -> /pmx/helpers/fast_lens_ext.py (vLLM worker extension,
importable in the engine process), mxf/.

Launch (MODAL_PROFILE=safety-sahan):
    DISAGG_GPU=B200:4 modal run --detach modal_rl_disagg.py::bench                    # throughput tables
    DISAGG_GPU=B200:4 modal run --detach modal_rl_disagg.py::main --n-rollout 1 --n-trainer 3 --total-steps 6
    DISAGG_GPU=B200:8 modal run --detach modal_rl_disagg.py::main --n-rollout 2 --n-trainer 6 --total-steps 400 \
        --extra-args "--cuda-graphs --run-name rl_everything_8x128_disagg --save-dir /data/ckpts_last5_v15_disagg"
Resume from a checkpoint: --extra-args "... --init-adapter <ckpt>/step_N --ref-adapter /data/sft_mix/last5_rp/final --step-offset N+1 --wandb-id <id>"
Set DISAGG_GPU (e.g. H200:4) at `modal run` time to pick the GPU request; the container's real GPU count
is what the launcher uses.
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent   # repo root (this launcher lives one level down)
app = modal.App("maemm-rl-disagg")
GPU = os.environ.get("DISAGG_GPU", "B200:4")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
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
    .pip_install("flash-linear-attention==0.5.2")   # fla + fla-core + einops; transformers>=4.45 already satisfied
    .pip_install("anthropic")                        # native Sonnet 5 judge for the inline extra evals (inline_extra_evals.JudgeClient)
)
# Hopper (H100/H200) trainer fix, opt-in at deploy time: DISAGG_TRAINER_TRITON=3.7.1. fla 0.5.2 raises on Triton 3.4-3.7.0 on
# Hopper ("produces incorrect results for gated chunk_bwd_dqkwg", fla #640); torch 2.10 pins triton 3.6.0, which vLLM also
# uses, so the newer Triton goes into its own dir and ONLY the HF trainer children see it (rl_disagg.run_launch spawn()).
TRAINER_TRITON = os.environ.get("DISAGG_TRAINER_TRITON", "")
if TRAINER_TRITON:
    # NOTE: this module is ALSO imported inside the container, where DISAGG_TRAINER_TRITON is not set -- bake the value
    # into the image env so _env() (which runs in the container) sees it (same bug class as the old N_GPU re-read).
    image = (image.run_commands(f"pip install --no-deps --target /opt/triton_{TRAINER_TRITON} triton=={TRAINER_TRITON}")
                  .env({"DISAGG_TRAINER_TRITON": TRAINER_TRITON}))
image = (
    image
    .add_local_file(REPO / "rl" / "rl.py", "/pmx/RL/rl_hf.py")
    .add_local_file(REPO / "rl" / "rl_disagg.py", "/pmx/RL/rl_disagg.py")
    .add_local_file(REPO / "rl" / "fast_lens_ext.py", "/pmx/helpers/fast_lens_ext.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "eval" / "eval_universal.py", "/pmx/eval/eval_universal.py")            # inline eval scoring
    .add_local_file(REPO / "eval" / "inline_extra_evals.py", "/pmx/RL/inline_extra_evals.py")          # autointerp/locality/WildChat/adversarial
    .add_local_file(REPO / "eval" / "snippet_locality.py", "/pmx/eval/snippet_locality.py")
    .add_local_file(REPO / "eval" / "autointerp_detection.py", "/pmx/eval/autointerp_detection.py")
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

POOL_DIR = "/data/banks/everything"
SFT_INIT = "/data/sft_mix/last5_rp/final"
CKPT_DIR = "/data/ckpts_disagg_dev"

# EXACT v15 production recipe (modal_rl_last5_v15.py TRAIN_ARGS) minus the flags rl_disagg ignores
TRAIN_ARGS = [
    "--bank-file", "vecs.f32",
    "--init-adapter", SFT_INIT,
    "--lr", "1e-5",
    "--reward-metric", "cosine",
    "--reward-scale", "1",
    "--len-penalty-start", "8",       # user (Sep 3): back to the original LP — active from 8 tokens
    "--len-penalty-per-tok", "0.00025",  # user: 0.00025 cosine per token (was 0.01 past 32 = EasyNLA hinge; "don't do this")
    "--no-gates",
    "--kl-coef", "0.01",
    "--adv-mode", "group",
    "--adam-eps", "1e-8",
    "--adam-betas", "0.9", "0.95",
    "--loss-agg", "seq",
    "--max-grad-norm", "1",
    "--reward-window-last", "5",
    "--reward-topk", "1",
    "--min-new-tokens", "8",
    "--max-new-tokens", "96",
    "--groups-per-step", "128",
    "--group-size", "8",
    "--score-batch", "128",
    "--ref-micro-batch", "32",
    "--vllm-gpu-mem", "0.85",
    "--inline-eval-every", "0",    # in-trainer eval OFF (user, Sep 3): checkpoints are evaluated by the separate 1-GPU daemon modal_eval_ckpt.py
    "--save-every", "0",
    "--save-steps", "25,40,60,90,130,200,300,450,675,1000",   # user: ~10 log-spaced ckpts to 1000; a separate 1-GPU job evals them
    "--warmup-steps", "10",                                      # user: 10-step linear LR warmup for stability
    "--save-dir", CKPT_DIR,
    "--run-name", "rl_everything_8x128_disagg_dev",
]


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers:/pmx/eval:/pmx/RL"
    if TRAINER_TRITON:
        env["DISAGG_TRAINER_PYTHONPATH"] = f"/opt/triton_{TRAINER_TRITON}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"
    env["HF_HOME"] = "/data/hf_cache"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs("/tmp/wandb", exist_ok=True)
    return env


def _stage():
    import shutil
    import time
    os.environ["HF_HOME"] = "/data/hf_cache"
    for p in (f"{POOL_DIR}/vecs.f32", f"{POOL_DIR}/records.jsonl", f"{POOL_DIR}/build_stats.json",
              f"{SFT_INIT}/adapter_model.safetensors", f"{SFT_INIT}/adapter_config.json"):
        assert os.path.exists(p), f"missing {p}"
    t0 = time.time()
    local_pool = "/root/pool"
    if not os.path.exists(local_pool):
        shutil.copytree(POOL_DIR, local_pool)
    print(f"[modal] pool staged to {local_pool} ({time.time() - t0:.0f}s)", flush=True)
    return local_pool


def _work_dir():
    """RAM-backed /dev/shm when it is big enough (adapter publish = 1-2 GB writes per step; the container
    overlay disk took 35 s per fp32 publish), else /tmp."""
    import shutil
    try:
        free = shutil.disk_usage("/dev/shm").free
    except Exception:  # noqa
        free = 0
    d = "/dev/shm/disagg" if free > 24 * 2**30 else "/tmp/disagg"
    print(f"[modal] work dir {d} (/dev/shm free {free / 2**30:.0f} GB)", flush=True)
    return d


def _run(cmd, env):
    """Run torchrun as its own process group and KILL it if this function is cancelled or errors —
    otherwise a Modal cancel only interrupts this Python thread and the trainer keeps running as a
    zombie in the warm container (happened 2026-09-03: a cancelled 8x128 leg trained on for 20 min)."""
    import os
    import signal
    import subprocess
    print("[modal] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
        return p.wait()
    finally:
        if p.poll() is None:
            print("[modal] terminating trainer process group (cancel/error)", flush=True)
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=20)
            except Exception:  # noqa
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:  # noqa
                    pass


def _collect(work="/tmp/disagg"):
    """Copy the json artefacts (bench tables, per-step trainer log, injection checks) to the volume."""
    import glob
    import shutil
    import time
    out = f"/data/disagg_runs/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(f"{work}/*.json"):
        shutil.copy(f, out)
    vol.commit()
    print(f"[modal] artefacts -> {out}: {sorted(os.listdir(out))}", flush=True)
    return out


@app.function(image=image, gpu=GPU, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb"),
                       modal.Secret.from_name("maemm-openrouter"),   # judge fallback
                       modal.Secret.from_name("maemm-anthropic")],   # native Sonnet 5 judge: ANTHROPIC_API_KEY + ANTHROPIC_WORKSPACE_ID
              timeout=24 * 3600)
def train(n_rollout: int = 1, n_trainer: int = 3, total_steps: int = 6, extra_args: str = "", no_wandb: bool = False):
    local_pool = _stage()
    args = list(TRAIN_ARGS)
    if no_wandb:
        args.append("--no-wandb")
    if extra_args:
        args += extra_args.split()
    work = _work_dir()
    cmd = ["python", "RL/rl_disagg.py", "--role", "launch", "--n-rollout", str(n_rollout), "--n-trainer", str(n_trainer),
           "--data-dir", local_pool, "--total-steps", str(total_steps), "--work-dir", work] + args
    rc = _run(cmd, _env())
    _collect(work)
    if rc != 0:
        raise RuntimeError(f"rl_disagg exited rc={rc}")


@app.function(image=image, gpu=GPU, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb")], timeout=4 * 3600)
def bench(n_rollout: int = 2, n_trainer: int = 2, extra_args: str = ""):
    """Trainer bench (micro-batch search + update time vs rollouts/rank) on GPUs [0,Y) and rollout bench
    (tok/s vs max_num_seqs x eager/graphs/stock-hook) on GPUs [Y,N), concurrently."""
    local_pool = _stage()
    args = list(TRAIN_ARGS) + ["--no-wandb"]
    if extra_args:
        args += extra_args.split()
    work = _work_dir()
    cmd = ["python", "RL/rl_disagg.py", "--role", "bench", "--n-rollout", str(n_rollout), "--n-trainer", str(n_trainer),
           "--data-dir", local_pool, "--work-dir", work] + args
    rc = _run(cmd, _env())
    _collect(work)
    if rc != 0:
        raise RuntimeError(f"bench exited rc={rc}")


@app.local_entrypoint()
def main(n_rollout: int = 1, n_trainer: int = 3, total_steps: int = 6, extra_args: str = "", no_wandb: bool = False):
    train.remote(n_rollout=n_rollout, n_trainer=n_trainer, total_steps=total_steps, extra_args=extra_args, no_wandb=no_wandb)


@app.local_entrypoint()
def run_bench(n_rollout: int = 2, n_trainer: int = 2, extra_args: str = ""):
    bench.remote(n_rollout=n_rollout, n_trainer=n_trainer, extra_args=extra_args)
