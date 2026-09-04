"""Which multi-GPU configs can we actually get, and how much memory per card?

Matters because the measured per-rank peak for this run is ~165 GiB at 128 sequences/rank. A 180 GiB
B200 reproduces the current box exactly; a 141 GiB H200 needs a smaller per-rank load; an 80 GiB
H100 leaves only ~26 GiB after the 27B weights and would force per-rank batches small enough to lose
most of the benefit of extra GPUs.
"""
import modal

app = modal.App("celeste-modlens-probe")
img = modal.Image.debian_slim().pip_install("torch==2.8.0")


def _report(tag):
    import subprocess
    out = subprocess.run(["nvidia-smi",
                          "--query-gpu=index,name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True)
    print("[%s]\n%s" % (tag, out.stdout.strip() or out.stderr.strip()), flush=True)
    return out.stdout.strip()


@app.function(image=img, gpu="B200:8", timeout=300)
def b200_8():
    return _report("B200:8")


@app.function(image=img, gpu="H200:8", timeout=300)
def h200_8():
    return _report("H200:8")


@app.function(image=img, gpu="H100:8", timeout=300)
def h100_8():
    return _report("H100:8")
