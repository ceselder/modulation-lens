"""Read out RL checkpoints on the RESERVED holdout activations, then judge them blind against the
warm start.

WHY THIS EXISTS. The RL reward is a learned surrogate (a frozen text->vector AR composed through J
and the affine), so a rising reward is not evidence the lens got better -- it is evidence the
surrogate is happier. This session already produced three cases where a proxy moved and the thing
it proxied did not (whitening/centring/affine bought FVE with no readability gain; a guessed
fluency floor "gated" 99.4% of good text). The only claim worth making is: on activations the run
NEVER trained on, does a human-legible judge prefer the RL readouts to the SFT readouts?

build_bank.py reserves 2048 rows (vecs_holdout.f32 + _meta.jsonl) for exactly this. Their meta
carries `label` (the span the model had just read) and `ctx`, which is what the judge scores
against.

  modal run rl/eval_holdout.py::main --ckpts "/vol/av_sft_4b/final,/vol/ckpts_modlens_scalerl400/step_100"
"""
import os, subprocess, modal

app = modal.App("modlens-eval-holdout")
vol = modal.Volume.from_name("celeste-modlens-vol")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "transformers==5.15.0", "peft", "accelerate",
                      "numpy", "safetensors", "flash-linear-attention", "einops")
         .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir("src", "/root/src")           # add_local_* must come last in the chain
         .add_local_file("rl/eval_holdout.py", "/root/eval_holdout.py"))

HOLDOUT = "/vol/rl_bank/vecs_holdout.f32"
HOLDOUT_META = "/vol/rl_bank/vecs_holdout_meta.jsonl"
PROMPT = "/vol/av_sft_4b/prompt.txt"
OUT = "/vol/eval_holdout"


@app.function(image=image, volumes={"/vol": vol}, gpu="B200", timeout=7200)
def readout(ckpt: str, n: int = 128, max_new: int = 96):
    os.makedirs(OUT, exist_ok=True)
    tag = ckpt.rstrip("/").replace("/vol/", "").replace("/", "_")
    out = f"{OUT}/{tag}.json"
    cmd = ["python", "/root/src/av_readout.py", "--ckpt", ckpt,
           "--probe-npy", HOLDOUT, "--probe-meta", HOLDOUT_META,
           "--prompt-file", PROMPT, "--n", str(n), "--max-new", str(max_new),
           "--temp", "0.0",                     # greedy: the readout must be deterministic to compare
           "--inject", "replace", "--layer", "42", "--out", out]
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root")
    vol.commit()
    print("[done]", out, flush=True)
    return out


@app.local_entrypoint()
def main(ckpts: str, n: int = 128):
    for r in readout.starmap([(c.strip(), n) for c in ckpts.split(",") if c.strip()]):
        print("wrote", r)
