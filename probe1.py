"""One GPU of each type: allocation is fast, so this answers AVAILABILITY quickly.
8-GPU requests can queue for a long time when the type is scarce, which is indistinguishable from
the type not existing -- this separates the two."""
import modal, subprocess
app = modal.App("celeste-modlens-probe1")
img = modal.Image.debian_slim()

def rep(tag):
    o = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                       capture_output=True, text=True)
    print("[%s] %s" % (tag, (o.stdout or o.stderr).strip()), flush=True)
    return (o.stdout or "").strip()

@app.function(image=img, gpu="B200", timeout=240)
def b200(): return rep("B200x1")

@app.function(image=img, gpu="H200", timeout=240)
def h200(): return rep("H200x1")

@app.function(image=img, gpu="H100", timeout=240)
def h100(): return rep("H100x1")
