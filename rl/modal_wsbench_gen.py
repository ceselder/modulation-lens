"""Run workspace-bench readout generation on MODAL (the CA-MTL route is down).

Both existing runners ship the checkpoint to CA-MTL to generate and judge back here; every CA-MTL
and EUR-IS node now refuses the key, so there is no route from a Modal checkpoint to the bench.
This is that route. It does NOT reimplement the contract -- it runs the bench's own
`modlens_wsbench_gen.py`, which already emits
    <out>/<family>/<label>/L042.jsonl   rows {label,family,layer,pos,token,samples}
    <out>/<family>/manifest.json        {model_id, layers, prompts[...]}
so the repo's deterministic scorers run unchanged afterwards, locally.

Mechanical families only (no judge, no API spend):
    multihop multilingual poetry typo association basic-readout
    multihop-mt multilingual-mt typo-mt basic-readout-mt          = 10 banks x 100 items

  modal run rl/modal_wsbench_gen.py::main --ckpt /vol/ckpts_modlens_kl0/step_25 --tag kl0_step25
"""
import os, subprocess, modal

app = modal.App("modlens-wsbench-gen")
vol = modal.Volume.from_name("celeste-modlens-vol")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "transformers==5.15.0", "peft", "accelerate", "numpy",
                      "safetensors", "flash-linear-attention", "einops", "pyyaml")
         .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir("/home/celeste/workspace-bench", "/root/bench", ignore=["*.log", ".git"])
         .add_local_dir("src", "/root/inv_src")
         .add_local_file("rl/modal_wsbench_gen.py", "/root/w.py"))

MECHANICAL = ("multihop,multilingual,poetry,typo,association,basic-readout,"
              "multihop-mt,multilingual-mt,typo-mt,basic-readout-mt")
OUT_VOL = "/vol/wsbench_readouts"


@app.function(image=image, volumes={"/vol": vol}, gpu="B200", timeout=21600)
def gen(ckpt: str, tag: str, families: str = MECHANICAL, max_new: int = 64,
        budget: int = 80, temp: float = 1.0, gen_batch: int = 64, limit: int = 0,
        prompt_file: str = "/vol/av_sft_4b/prompt.txt"):
    out = f"{OUT_VOL}/{tag}"
    os.makedirs(out, exist_ok=True)
    cmd = ["python", "/root/bench/modlens_wsbench_gen.py",
           "--adapter", ckpt,      # the bench calls it --adapter, not --ckpt
           "--inv-src", "/root/inv_src",
           "--bench-root", "/root/bench",
           "--out-root", out,
           "--families", families,
           "--layer", "42",
           "--max-new", str(max_new),
           "--budget-tokens", str(budget),
           "--temp", str(temp),
           "--gen-batch", str(gen_batch),
           "--inject", "replace",          # the mode this lens was trained with
           "--prompt-file", prompt_file]   # NOT optional: a different prompt is off-distribution
    if limit:
        cmd += ["--limit", str(limit)]
    print("[cmd]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd="/root/bench", env={**os.environ, "PYTHONPATH": "/root/bench"})
    vol.commit()
    fams = sorted(os.listdir(out)) if os.path.isdir(out) else []
    print("[out] %s -> %d family dirs: %s" % (out, len(fams), fams), flush=True)
    return {"rc": r.returncode, "out": out, "families": fams}


@app.local_entrypoint()
def main(ckpt: str = "/vol/ckpts_modlens_kl0/step_25", tag: str = "kl0_step25",
         families: str = MECHANICAL, limit: int = 0, max_new: int = 64):
    print(gen.remote(ckpt=ckpt, tag=tag, families=families, limit=limit, max_new=max_new))
