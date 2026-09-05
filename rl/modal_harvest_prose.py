"""Sharded corpus harvest: scale the RL target bank past one epoch.

WHY. 100 steps x 4096 rollouts = 409,600 against a 497,952-row bank is 0.82 EPOCHS -- the 100-step
runs never finished a single pass, so "train longer" on this bank would mean re-visiting targets
rather than seeing new ones. This project's rule is explicit: scaling a run means MORE DATA and one
epoch, not more epochs on a fixed set. This fans harvest_prose.py across N GPUs over disjoint slices
of the same deterministic Ultra-FineWeb stream (--skip-docs), then build_bank consumes the glob.

  modal run rl/modal_harvest_prose.py::main --shards 8 --per-shard 500000
  -> /vol/data/prose_L42_shards/shard_XX.parquet  (8 x 500k = 4.0M rows, ~9.8 epochs of headroom)
"""
import os, subprocess, modal

app = modal.App("modlens-harvest-prose")
vol = modal.Volume.from_name("celeste-modlens-vol")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "transformers==5.15.0", "accelerate", "numpy", "pyarrow",
                      "datasets", "safetensors", "flash-linear-attention", "einops", "peft")
         .env({"HF_HOME": "/vol/.hf_home", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir("src", "/root/src")
         .add_local_dir("mxf", "/root/mxf")
         .add_local_file("rl/modal_harvest_prose.py", "/root/h.py"))

OUT_DIR = "/vol/data/prose_L42_shards"
# ~6 positions per document, so a shard of P rows consumes ~P/6 documents. Stride must EXCEED that
# or shards overlap; 3x headroom absorbs documents skipped for being too short.
DOCS_PER_ROW = 1.0 / 6.0
STRIDE_SAFETY = 3.0


@app.function(image=image, volumes={"/vol": vol}, gpu="B200", timeout=86400,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def harvest(shard: int, per_shard: int, layer: int = 42, n_shards: int = 8):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/shard_{shard:02d}.parquet"
    if os.path.exists(out):
        # GUARDED: a preempted container leaves a 4-byte stub holding only the PAR1 magic, and
        # reading it raises ArrowInvalid ("file size is 4 bytes, smaller than the minimum file
        # footer"). An unreadable or short file must mean REDO, never crash.
        import pyarrow.parquet as pq
        try:
            n = pq.ParquetFile(out).metadata.num_rows
        except Exception as e:
            print(f"[redo] shard {shard}: {out} unreadable ({str(e)[:80]}) -- removing", flush=True)
            os.remove(out); n = 0
        if n >= per_shard * 0.98:
            print(f"[skip] shard {shard} already has {n:,} rows", flush=True)
            return {"shard": shard, "rows": n, "skipped": True}
        if n:
            print(f"[redo] shard {shard} only has {n:,} of {per_shard:,}", flush=True)
            os.remove(out)
    cmd = ["python", "/root/src/harvest_prose.py", "--n", str(per_shard), "--layer", str(layer),
           "--num-shards", str(n_shards), "--shard-idx", str(shard),
           "--seed", str(shard), "--out", out]
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    vol.commit()
    import pyarrow.parquet as pq
    n = pq.ParquetFile(out).metadata.num_rows
    print(f"[done] shard {shard}: {n:,} rows -> {out}", flush=True)
    return {"shard": shard, "rows": n, "skipped": False}


@app.local_entrypoint()
def main(shards: int = 8, per_shard: int = 500000, layer: int = 42):
    tot = 0
    for r in harvest.starmap([(k, per_shard, layer, shards) for k in range(shards)]):
        print(r)
        tot += r["rows"]
    print(f"\nTOTAL {tot:,} rows across {shards} shards -> {OUT_DIR}")
    print(f"  at 4096 rollouts/step that is {tot / 4096:.0f} steps of ONE epoch")
