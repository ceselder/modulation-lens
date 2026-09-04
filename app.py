"""Modulation-lens training on Modal, to get past the 4-GPU per-rank memory ceiling.

Why Modal at all: the binding constraint on the current box is PER-RANK memory (~128 sequences at
micro 4, measured 165 GiB of 178). rollouts/step = batch x group, so reaching batch 128 x group 16
(2048 rollouts) needs 16 ranks. The local SLURM caps --gres at 8 on one node and does not do
multi-node, so extra ranks have to come from somewhere else.

Design notes:
  * The image pins torch/transformers/peft to the versions the training box runs, because inv_core's
    grid read and the GDN/fla kernels are the fragile parts and a version drift there produces
    silently different activations rather than an error.
  * /workspace/.hf_home is SYMLINKED to the volume, so inv_core's hardcoded J-lens glob resolves
    without editing the file -- the Modal run then executes byte-identical code to the B200 box.
  * Weights and data live on a Volume, so they are downloaded once rather than per cold start.
  * NEVER reuse another fellow's app or volume name: everything here is celeste-modlens-*.
"""
import os
import subprocess

import modal
import modal.experimental  # the clustered decorator is resolved at import time

APP = "celeste-modlens"
VOL = modal.Volume.from_name("celeste-modlens-vol", create_if_missing=True)
GPU = os.environ.get("MODLENS_GPU", "B200:8")
NODES = int(os.environ.get("MODLENS_NODES", "2"))   # 2 containers x 8 = 16 ranks

app = modal.App(APP)

img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.5.4",
        "peft==0.19.1",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "pyarrow",
        "numpy",
        "pandas",
        "wandb",
        "einops",
        "huggingface_hub[hf_transfer]",
        "flash-linear-attention",
    )
    # HF_HOME belongs in the IMAGE env, not in the function body: huggingface_hub resolves it into
    # constants.HF_HUB_CACHE at IMPORT time, so setting os.environ after the import silently
    # downloads to the container's ephemeral cache and the volume stays empty -- with exit code 0.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "HF_HOME": "/vol/.hf_home"})
    .add_local_dir("/home/celeste/modlens_modal/src", "/root/src", copy=True)
)


def _link_hf():
    """inv_core globs a hardcoded /workspace/.hf_home path for the J-lens; point it at the volume."""
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")


@app.function(image=img, volumes={"/vol": VOL}, timeout=7200,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def fetch_weights():
    """One-time: pull the base model and the J-lens onto the volume."""
    from huggingface_hub import snapshot_download
    CACHE = "/vol/.hf_home/hub"
    for repo in ("Qwen/Qwen3.6-27B", "camilablank/workspace-lenses"):
        print("[fetch] %s" % repo, flush=True)
        # cache_dir explicitly: independent of when HF_HOME was read
        snapshot_download(repo_id=repo, cache_dir=CACHE,
                          token=os.environ.get("HF_TOKEN") or None)
    VOL.commit()
    # VERIFY rather than trust the exit code: the previous run exited 0 having written the weights
    # to an ephemeral directory.
    du = subprocess.run(["du", "-sh", CACHE], capture_output=True, text=True).stdout.strip()
    ls = subprocess.run(["ls", CACHE], capture_output=True, text=True).stdout.split()
    jl = subprocess.run(["bash", "-lc",
                         "ls %s/models--camilablank--workspace-lenses/snapshots/*/qwen3.6-27b/"
                         "j-lens/lens.pt 2>/dev/null" % CACHE],
                        capture_output=True, text=True).stdout.strip()
    print("[fetch] cache: %s" % du, flush=True)
    print("[fetch] repos: %s" % ls, flush=True)
    print("[fetch] j-lens present: %s" % (jl or "MISSING"), flush=True)
    assert any("Qwen3.6-27B" in x for x in ls), "base model missing from the volume"
    assert jl, "j-lens missing from the volume"
    print("FETCH_VERIFIED", flush=True)
    return du


@app.function(image=img, volumes={"/vol": VOL}, gpu=GPU, timeout=86000,
              secrets=[modal.Secret.from_dict({
                  "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                  "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})])
def train(args: str, nproc: int = 8):
    """Run inv_train.py under torchrun with `args` (a single shell-style string)."""
    _link_hf()
    env = dict(os.environ,
               HF_HOME="/vol/.hf_home", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader"], check=False)
    cmd = ["torchrun", "--nproc_per_node", str(nproc), "/root/src/inv_train.py"] + args.split()
    print("[train] %s" % " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env)
    VOL.commit()
    print("[train] exit %d" % p.returncode, flush=True)
    return p.returncode


@app.function(image=img, volumes={"/vol": VOL}, gpu=GPU, timeout=900)
def gpu_report():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    print(out, flush=True)
    return out

@app.function(image=img, volumes={"/vol": VOL}, gpu=GPU, timeout=5400,
              secrets=[modal.Secret.from_dict({
                  "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                  "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})])
def smoke(nproc: int = 8):
    """Prove this environment reproduces the training box BEFORE paying for a long run.

    The fingerprint is |PMU| and the grid signature. PMU is the mean of the 36-cell grid read over
    filler phrases and the signature hashes the TOKENIZED cells, so together they cover the
    tokenizer, the chat template, the injection marker, the layer-42 hook and the J-lens. If both
    match the local box (|PMU| 57.79, sig db4a6b8ee6) the environments are equivalent and Modal
    numbers are comparable with everything measured locally. If they differ, any Modal result would
    be silently incomparable -- which is far worse than a crash.
    """
    _link_hf()
    env = dict(os.environ, HF_HOME="/vol/.hf_home", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               WANDB_MODE="disabled")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader"], check=False)
    args = ("--mode rl --inject karvonen --bullets 4 --bullet-max-tok 10 --lam-div 0 "
            "--policy /vol/ckpts/sft_nnols4_fresh/final "
            "--data /vol/data/prose_L42_500k.parquet --n-pool 500000 "
            "--whitener /vol/data/natural_whitener_jspace.npz "
            "--probe-npy /vol/data/holdout_blogpost.npy --probe-meta /vol/data/holdout_blogpost.jsonl "
            "--whiten 0 --n-carriers 6 --batch %d --group 8 --micro 4 "
            "--kl-beta 0 --read-batch 256 --lam-text 0 --max-new 128 --min-words 4 "
            "--lr 1e-5 --steps 3 --eval-every 999 --save-every 999 "
            "--save-dir /vol/ckpts/_smoke --wandb-name modal_smoke") % (nproc * 16)
    cmd = ["torchrun", "--nproc_per_node", str(nproc), "/root/src/inv_train.py"] + args.split()
    print("[smoke] %s" % " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = (p.stdout or "")[-4000:] + (p.stderr or "")[-4000:]
    print(tail, flush=True)
    import re
    m = re.search(r"\|PMU\| ([\d.]+) \[grid ([a-f0-9]+)\]", tail)
    if not m:
        print("SMOKE_NO_FINGERPRINT (exit %d)" % p.returncode, flush=True)
        return {"ok": False, "rc": p.returncode}
    pmu, sig = float(m.group(1)), m.group(2)
    ok = abs(pmu - 57.79) < 0.02 and sig == "db4a6b8ee6"
    print("[smoke] |PMU| %.2f sig %s  -> %s local (57.79 / db4a6b8ee6)"
          % (pmu, sig, "MATCHES" if ok else "DIFFERS FROM"), flush=True)
    print("SMOKE_MATCH" if ok else "SMOKE_MISMATCH", flush=True)
    return {"ok": ok, "pmu": pmu, "sig": sig, "rc": p.returncode}

@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:8", timeout=86000,
              secrets=[modal.Secret.from_dict({
                  "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                  "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})])
@modal.experimental.clustered(size=NODES)   # rdma unavailable on this workspace
def train_clustered(args: str):
    """MODLENS_NODES containers x 8 B200. At NODES=2 that is 16 ranks for batch 128 x group 16 =
    2048 rollouts/step; at NODES=4, 32 ranks for batch 256 x group 16.

    Per-rank memory is the binding constraint, measured: 128 sequences fits at 165 GiB of 183, 256
    sequences reaches 178 GiB (26 MiB from OOM), so ~104 MiB per extra sequence on a ~152 GiB base.
    4096 rollouts therefore needs 4096/128 = 32 ranks; 512 sequences on 8 ranks extrapolates to
    ~204 GiB and cannot fit. RDMA is not enabled on this workspace, which is acceptable here: only
    LoRA gradients are all-reduced (~1.3 GB across four nodes, well under a second) against a ~131 s
    step, so interconnect is not the bottleneck -- the 36-cell grid read is.
    """
    from modal.experimental import get_cluster_info
    ci = get_cluster_info()
    rank, ips = ci.rank, ci.container_ips
    _link_hf()
    env = dict(os.environ, HF_HOME="/vol/.hf_home", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               NCCL_DEBUG="WARN")
    print("[cluster] node %d/%d master=%s" % (rank, len(ips), ips[0]), flush=True)
    cmd = ["torchrun",
           "--nnodes", str(len(ips)), "--node_rank", str(rank),
           "--master_addr", ips[0], "--master_port", "29900",
           "--nproc_per_node", "8",
           "/root/src/inv_train.py"] + args.split()
    print("[cluster] %s" % " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env)
    if rank == 0:
        VOL.commit()
    print("[cluster] node %d exit %d" % (rank, p.returncode), flush=True)
    return p.returncode
