"""Benchmark vllm-metamodel against the HF-transformers measurement loop.

The current pass runs plain HF forwards with a layer-42 hook at batch 48 and sustains ~220
cell-reads/s on a B200 -- about 14k tokens/s, i.e. 12-18% MFU for an 18B-active model. The
optimisation effort went into prefix caching (rejected: numerically wrong on GDN), batch size
(backwards -- 256 and 512 were SLOWER), layer truncation (1.24x) and skipping lm_head (no gain),
all <=1.5x, while the framework itself was never questioned. That is plausibly the 10x.

vllm-metamodel (ceselder/vllm-metamodel) is a drop-in vllm-lens fork that indexes the steering hook
instead of looping over the batch with per-request GPU syncs; reported 37.8x at B=2048 on this exact
model. Our pass READS rather than injects, so that specific speedup does not apply -- what should
carry over is vLLM's prefill path, continuous batching and CUDA graphs, plus dropping the
length-bucketing the HF loop needs (vLLM takes ragged batches natively).

Capture API, verified upstream against HF to <1e-2 mean abs diff:
    SamplingParams(max_tokens=1, extra_args={"output_residual_stream": [42]})
    out.activations["residual_stream"]
"""
import os

import modal

app = modal.App("celeste-bench-vllm-metamodel")
VOL = modal.Volume.from_name("celeste-modlens-vol")
# CUDA DEVEL base, not debian_slim: vLLM needs nvcc/CUDA_HOME at RUNTIME (its
# determine_available_memory path shells out to a CUDA utility) and fails with
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist". This is the same
# missing-toolkit problem that broke the causal-conv1d build earlier.
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .apt_install("git", "build-essential")
       .pip_install("torch==2.8.0", "numpy", "pyarrow", "huggingface_hub[hf_transfer]")
       .pip_install("git+https://github.com/ceselder/vllm-metamodel")
       .pip_install("transformers==5.5.4", "flash-linear-attention", "einops")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "CUDA_HOME": "/usr/local/cuda",
             # The fork's fast path is opt-in: "Without the variable, behaviour is exactly
             # 1.1.0's (eager forced)". With it, prefill stays eager (so the capture hooks fire at
             # prompt positions -- which is exactly what we read) while decode runs as graph
             # replays. Our workload is 63 tokens of prefill and one decode step, so graphs should
             # buy us almost nothing; setting it anyway so the comparison is against the intended
             # configuration rather than a crippled one.
             "VLLM_LENS_CUDA_GRAPHS": "1",
             "TOKENIZERS_PARALLELISM": "false",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       .add_local_dir("/home/celeste/modlens_modal/src", "/root/src", copy=True))

WORKER = r'''
import os, sys, time
import numpy as np, torch


def main():
    sys.path.insert(0, "/root/src")
    import inv_core as C
    from transformers import AutoTokenizer
    # Do NOT call register() before LLM(): it patches create_engine_config, and vLLM's plugin
    # loader then calls register() AGAIN during LLM.__init__, so the patch wraps itself and the
    # constructor recurses (RecursionError at llm.py:343 -- observed twice). Let vLLM register, then
    # patch ONLY LLM.generate afterwards if the client process still lacks it, which is what makes
    # `activations` appear on RequestOutput. Patching generate post-construction is safe: the
    # engine config is already built, so there is nothing left to double-patch.
    from vllm import LLM, SamplingParams

    LAYER = 42
    NPH = int(os.environ.get("NPHRASE", "600"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    G = C.Grid(tok, C.TEMPLATES_RECOVERED[:4], C.CARRIERS_RECOVERED[:4], LAYER,
               torch.eye(1), "cpu")
    cells = [(c, t) for c in range(G.n_car) for t in range(G.n_tpl)]
    print("[grid] %dt x %dc = %d cells" % (G.n_tpl, G.n_car, len(cells)), flush=True)

    base = ["the color of the grass", "Valley girl", "Add some drama",
            "inorganic chemical compounds", "tons of earth and debris", "make it funny",
            "Sea chest", "parts of speech"]
    phrases = (base * (NPH // len(base) + 1))[:NPH]

    def build(n_phrase):
        prompts, ncars = [], []
        for ph in phrases[:n_phrase]:
            mid = tok(ph, add_special_tokens=False).input_ids[:24]
            for ci, ti in cells:
                cell = G.cells[ci][ti]
                prompts.append(tok.decode(cell["pre"] + mid + cell["post"]))
                ncars.append(cell["ncar"])
        return prompts, ncars

    t0 = time.time()
    # max_num_seqs <= 1024: the README documents a vLLM bug where Qwen3.5/3.6 hybrid GDN models
    # die in graph warm-up at 2048 because the packed decode kernel launches a batch x 48-head grid
    # against CUDA's 65,535 grid-dim limit. max_num_batched_tokens above longest prompt x
    # concurrency, per the chunked-prefill warning.
    llm = LLM(model="Qwen/Qwen3.6-27B", dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=512, max_num_seqs=1024, max_num_batched_tokens=16384,
              # enable_prefix_caching MUST be off. Our 16 cell prompts share a byte-identical
              # `pre` segment, so vLLM would reuse cached prefix state -- exactly the optimisation
              # I built by hand and had to REJECT: on this model a 63-token sequence and a
              # 33-token prefix + 30-token continuation give different results (cosine 0.930, not
              # 1.0), because GDN's chunked kernel moves its chunk boundaries. Reusing a prefix
              # would corrupt every captured vector in a way no throughput number reveals.
              enable_prefix_caching=False)
    print("[vllm] engine up in %.0fs" % (time.time() - t0), flush=True)
    import vllm_lens._activations_plugin as _ap
    if getattr(_ap, "_original_llm_generate", None) is None:
        _ap._original_llm_generate = LLM.generate
        LLM.generate = _ap._patched_llm_generate
        print("[plugin] LLM.generate patched post-construction (client process lacked it)", flush=True)
    else:
        print("[plugin] LLM.generate already patched by vLLM's plugin loader", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1,
                        extra_args={"output_residual_stream": [LAYER]})

    print("\n%-24s %14s" % ("configuration", "cell-reads/s"), flush=True)
    best = 0.0
    for nph in (64, 256, 600):
        prompts, ncars = build(min(nph, NPH))
        torch.cuda.synchronize(); t = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize(); dt = time.time() - t
        rate = len(prompts) / dt
        best = max(best, rate)
        print("%-24s %14.1f   (%d prompts in %.1fs)"
              % ("vllm, %d spans" % nph, rate, len(prompts), dt), flush=True)
        if nph == 64:
            if not hasattr(outs[0], "activations"):
                o = outs[0]
                attrs = [a for a in dir(o) if not a.startswith("_")]
                print("   CAPTURE MISSING: RequestOutput has no .activations. The timing above is "
                      "therefore prefill WITHOUT capture (a floor on the real cost).", flush=True)
                print("   RequestOutput attrs: %s" % attrs, flush=True)
                print("   sampling extra_args round-tripped: %r"
                      % getattr(getattr(o, "sampling_params", None), "extra_args", "n/a"), flush=True)
                continue
            v = outs[0].activations["residual_stream"]
            print("   capture shape %s dtype %s -> mean-pool last %d positions"
                  % (tuple(v.shape), v.dtype, ncars[0]), flush=True)

    # correctness: a faster number is worthless if the activations differ. Compare a handful of
    # captured vectors against a plain HF forward on the same prompts.
    try:
        import torch as _t
        from transformers import AutoModelForCausalLM
        prompts, ncars = build(4)
        outs = llm.generate(prompts[:8], sp, use_tqdm=False)
        if hasattr(outs[0], "activations"):
            hf = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3.6-27B", dtype=_t.bfloat16).to("cuda").eval()
            worst = 1.0
            for o, pr, nc in zip(outs, prompts[:8], ncars[:8]):
                ids = _t.tensor([tok(pr, add_special_tokens=False).input_ids], device="cuda")
                with _t.no_grad():
                    h = hf(ids, output_hidden_states=True, use_cache=False
                           ).hidden_states[LAYER + 1][0].float()
                a = h[-nc:].mean(0)
                b = o.activations["residual_stream"][0, -nc:, :].float().mean(0).to("cuda")
                c = float(_t.nn.functional.cosine_similarity(a[None], b[None])[0])
                worst = min(worst, c)
            print("\n[verify] worst cosine vLLM vs HF over 8 prompts: %.6f %s"
                  % (worst, "OK" if worst > 0.999 else "<-- MISMATCH, do not use"), flush=True)
            del hf
    except Exception as e:
        print("\n[verify] could not run the HF cross-check: %s" % str(e)[:120], flush=True)

    print("\nHF baseline, same work: 220.0 cell-reads/s (measured, batch 48, 43 layers)", flush=True)
    if best:
        print("speedup %.1fx  |  11.6M spans x 16 cells: %.1f GPU-h vs %.1f GPU-h on HF"
              % (best / 220.0, 11.6e6 * 16 / best / 3600, 11.6e6 * 16 / 220.0 / 3600), flush=True)
    print("BENCH_VLLM_DONE", flush=True)


if __name__ == "__main__":
    main()
'''


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=5400,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def bench(nphrase: int = 600):
    import subprocess
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")
    open("/root/w.py", "w").write(WORKER)
    return subprocess.run(["python", "/root/w.py"],
                          env=dict(os.environ, NPHRASE=str(nphrase))).returncode
