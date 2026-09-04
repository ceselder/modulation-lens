"""Can causal-conv1d be built in a Modal image? It is the GDN fast path, worth ~1.8x.

The plain build fails with `NameError: name 'bare_metal_version' is not defined`, which is
causal-conv1d's setup.py failing to parse `nvcc --version` because debian_slim ships no CUDA
toolkit. Two candidate fixes, tested separately so we learn which one works rather than stacking
both and not knowing:

  A. pip-provide nvcc (nvidia-cuda-nvcc-cu12) and point CUDA_HOME at it  -- small image
  B. build from a nvidia/cuda devel base image                          -- large but reliable
"""
import modal

app = modal.App("celeste-test-ccv1d")

# --- A: pip-provided nvcc ---
img_a = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("build-essential", "git")
         .pip_install("torch==2.8.0", "packaging", "ninja", "setuptools", "wheel")
         .pip_install("nvidia-cuda-nvcc-cu12==12.8.93")
         .run_commands(
             "export CUDA_HOME=$(python -c \"import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))\") && "
             "export PATH=$CUDA_HOME/bin:$PATH && "
             "nvcc --version && "
             "CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation causal-conv1d",
             gpu="B200:1"))   # nvcc needs to know the arch; a GPU present makes torch report it

# --- B: CUDA devel base ---
img_b = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .apt_install("build-essential", "git")
         .pip_install("torch==2.8.0", "packaging", "ninja", "setuptools", "wheel")
         .run_commands("CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation causal-conv1d",
                       gpu="B200:1"))


@app.function(image=img_a, gpu="B200:1", timeout=3600)
def variant_a():
    import causal_conv1d
    print("A OK: causal_conv1d", getattr(causal_conv1d, "__version__", "?"))
    return "A_OK"


@app.function(image=img_b, gpu="B200:1", timeout=3600)
def variant_b():
    import causal_conv1d
    print("B OK: causal_conv1d", getattr(causal_conv1d, "__version__", "?"))
    return "B_OK"
