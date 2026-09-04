"""Do the random-acknowledging read templates capture a phrase's target better?

Readouts hedge when the policy cannot pin content down: 3.0% of emitted lines contain "random"
(concentrated on a few activations) and 20.4% retreat to describing the medium ("a snippet from a
conversation transcript"). The proposed fix tells the READER that the phrase may sound arbitrary, so
the policy no longer has to spend its budget apologising.

Before training anything on it, test whether the acknowledging grid is simply a better read: take
ONE set of phrases and ONE set of target activations, and score them through both grids. The target
(activation projected to J-space, minus the pool mean) is grid-independent, so this is a fair
comparison; each grid gets its OWN PMU, since PMU is that grid's own filler mean and reusing the
wrong one scores everything against the wrong centre.

Reports, per grid: mean cosine to target, and the same restricted to the phrases that contain a
hedge ("random", "maybe", "snippet/transcript"). If the acknowledgement helps at all it should help
most there.
"""
import os
import subprocess

import modal

app = modal.App("celeste-modlens-abgrid")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .apt_install("git")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "peft==0.19.1", "accelerate",
                    "safetensors", "sentencepiece", "pyarrow", "numpy",
                    "huggingface_hub[hf_transfer]", "einops", "flash-linear-attention")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       .add_local_dir("/home/celeste/modlens_modal/src", "/root/src", copy=True))

WORKER = r'''
import json, os, re, sys
import numpy as np, torch
sys.path.insert(0, "/root/src")
import inv_core as C
import pyarrow.parquet as pq
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ["AB_CKPT"]
NACT = int(os.environ.get("AB_NACT", "64"))
dev  = "cuda"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = base.model
INJ, LEFT, RIGHT = C.marker_ids(tok)
HOOK, L42 = {"vec": None, "ids": None}, {}
inner.register_forward_pre_hook(
    lambda m, a, kw: HOOK.__setitem__("ids", kw.get("input_ids") if kw.get("input_ids") is not None
                                      else (a[0] if a else None)), with_kwargs=True)
def _inj(m, a, out):
    r = out[0] if isinstance(out, tuple) else out
    ids, vec = HOOK["ids"], HOOK["vec"]
    if vec is None or ids is None or tuple(ids.shape) != tuple(r.shape[:-1]): return out
    if not bool((ids == INJ).any()): return out
    n = C.inject_at_marker(ids, r, vec, INJ, LEFT, RIGHT, mode="karvonen")
    return (n,) + tuple(out[1:]) if isinstance(out, tuple) else n
inner.layers[1].register_forward_hook(_inj)
inner.layers[42].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))

J = C.load_jlens(42, dev)
model = PeftModel.from_pretrained(base, CKPT, adapter_name="m").eval()
model.set_adapter("m")

JOB = open("/vol/ckpts/prompt.txt").read()
PIDS = torch.tensor(tok.encode(tok.apply_chat_template(
    [{"role": "user", "content": JOB}], tokenize=False, add_generation_prompt=True,
    enable_thinking=False), add_special_tokens=False), device=dev)
PLEN = PIDS.numel()

# targets: activations, centred by the training pool's mean, projected to J-space (grid-independent)
acc = []
for b in pq.ParquetFile("/vol/data/prose_L42_500k.parquet").iter_batches(
        batch_size=8192, columns=["activation_vector", "label"]):
    d = b.to_pydict()
    acc.append(np.asarray(d["activation_vector"], dtype="float32"))
    if sum(len(x) for x in acc) >= 100000: break
V = np.concatenate(acc); del acc
mu = torch.from_numpy(V.mean(0)).to(dev)
ACT = torch.from_numpy(np.load("/vol/data/holdout_fresh.npy").astype("float32"))[:NACT]

# ONE set of phrases, generated once, scored through both grids
with torch.no_grad():
    HOOK["vec"] = ACT.to(dev).float()
    try:
        g = model.generate(input_ids=PIDS.unsqueeze(0).expand(NACT, -1).contiguous(),
                           attention_mask=torch.ones(NACT, PLEN, device=dev, dtype=torch.long),
                           do_sample=False, max_new_tokens=39, pad_token_id=248046)
    finally:
        HOOK["vec"] = None
_BUL = re.compile(r"^\s*(?:[*•\-–]|\d+[.)])\s+")
phrases = []
for t in tok.batch_decode(g[:, PLEN:], skip_special_tokens=True):
    first = next((l for l in t.splitlines() if l.strip()), t)
    phrases.append(_BUL.sub("", first).strip() or "the")
print("[ab] %d phrases, e.g. %r" % (len(phrases), phrases[:3]), flush=True)

HEDGE = re.compile(r"\b(random|maybe|perhaps|possibly|snippet|transcript|corpus|something)\b", re.I)
hedged = [i for i, p_ in enumerate(phrases) if HEDGE.search(p_)]
print("[ab] phrases containing a hedge: %d of %d" % (len(hedged), len(phrases)), flush=True)

res = {}
for name, tpls in (("recovered", C.TEMPLATES_RECOVERED), ("random_ack", C.TEMPLATES_RANDOM_ACK)):
    grid = C.Grid(tok, tpls, C.CARRIERS_RECOVERED[:6], 42, J, dev)
    with torch.no_grad(), model.disable_adapter():
        pmu = grid.prompt_mean(model, L42, n=64, seed=0)
        cv = grid.read_all(model, list(dict.fromkeys(phrases)), L42, max_tok=16, batch=192)
    cos = []
    for i, ph in enumerate(phrases):
        v = cv[ph] - pmu
        t = (ACT[i].to(dev) @ J.T) - (mu @ J.T)
        t = t / t.norm().clamp(min=1e-8)
        cos.append(float((v @ t) / v.norm().clamp(min=1e-8)))
    res[name] = {"sig": grid.sig()[:10], "pmu_norm": float(pmu.norm()),
                 "mean_cos": float(np.mean(cos)),
                 "sem": float(np.std(cos, ddof=1) / np.sqrt(len(cos))),
                 "mean_cos_hedged": float(np.mean([cos[i] for i in hedged])) if hedged else None,
                 "mean_cos_clean": float(np.mean([c for i, c in enumerate(cos) if i not in hedged])),
                 "cos": cos}
    print("[ab] %-11s sig %s |PMU| %6.2f  mean cos %.4f +-%.4f  hedged %s  clean %.4f"
          % (name, res[name]["sig"], res[name]["pmu_norm"], res[name]["mean_cos"],
             res[name]["sem"],
             ("%.4f" % res[name]["mean_cos_hedged"]) if hedged else "n/a",
             res[name]["mean_cos_clean"]), flush=True)

a, b_ = res["recovered"], res["random_ack"]
d = [y - x for x, y in zip(a["cos"], b_["cos"])]
import statistics as st
sem_d = st.stdev(d) / (len(d) ** 0.5)
print("\n[ab] PAIRED difference (random_ack - recovered): %+.4f +-%.4f  t = %+.2f"
      % (st.mean(d), sem_d, st.mean(d) / sem_d), flush=True)
if hedged:
    dh = [d[i] for i in hedged]
    print("[ab]   on hedged phrases only: %+.4f (n=%d)" % (st.mean(dh), len(dh)), flush=True)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "cos"} for k, v in res.items()},
          open("/vol/results_abgrid.json", "w"), indent=1)
print("AB_GRID_DONE", flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=5400,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def ab(ckpt: str = "/vol/ckpts/rl_g16_micro8/iter_000025", nact: int = 64):
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")
    open("/root/worker.py", "w").write(WORKER)
    env = dict(os.environ, AB_CKPT=ckpt, AB_NACT=str(nact))
    p = subprocess.run(["python", "/root/worker.py"], env=env)
    VOL.commit()
    return p.returncode
