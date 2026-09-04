"""Fixed-batch replay: is the gradient-norm growth in the MODEL or in the ROLLOUT DISTRIBUTION?

Pre-clip gradient norm grows ~6x over 350 GRPO steps while LoRA weight norms move 1%. Two
hypotheses survive the arithmetic:

  (A) cross-group coherence / rollout-distribution shift -- the 64 per-group gradients stop
      cancelling as the policy converges on shared directions (length, format). 5.8x fits inside the
      sqrt(64)=8x budget with NO per-sample quantity growing.
  (B) something in the model grows -- activation scale, factor alignment, or falling p(sampled) which
      inflates the per-token logit gradient norm ~ |adv|*sqrt(2)*(1-p_s).

They are separated by holding the DATA fixed. Generate one batch of rollouts ONCE with the earliest
checkpoint, freeze those exact (activation, rollout, advantage) triples, then recompute the gradient
norm with each later checkpoint on that same frozen batch:

    norm stays flat  -> the growth lives in the rollout distribution   (A)
    norm reproduces ~6x -> the growth lives in the model               (B)

Also logs mean per-token rollout logprob per checkpoint, which is the direct test of the p_s channel,
and per-module gradient norms, whose profile distinguishes advantage/coherence (uniform) from
activation-scale growth (concentrated in o_proj and down_proj -- the only modules whose inputs are
not post-RMSNorm with frozen gains).
"""
import os
import re
import subprocess

import modal
import modal.experimental

app = modal.App("celeste-modlens-replay")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .apt_install("git")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "peft==0.19.1", "accelerate",
                    "safetensors", "sentencepiece", "pyarrow", "numpy",
                    "huggingface_hub[hf_transfer]", "einops", "flash-linear-attention")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       .add_local_dir("/home/celeste/modlens_modal/src", "/root/src", copy=True)
       .add_local_file(__file__, "/root/replay_driver.py", copy=True))

WORKER = r'''
import json, os, re, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/root/src")
import inv_core as C
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

RUN   = os.environ["REPLAY_RUN"]
CKPTS = os.environ["REPLAY_CKPTS"].split(",")
NACT  = int(os.environ.get("REPLAY_NACT", "16"))
GROUP = int(os.environ.get("REPLAY_GROUP", "16"))
dev = "cuda"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner = base.model
INJ, LEFT, RIGHT = C.marker_ids(tok)
HOOK = {"vec": None, "ids": None}
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

JOB = open("/vol/ckpts/prompt.txt").read()
PIDS = torch.tensor(tok.encode(tok.apply_chat_template(
    [{"role": "user", "content": JOB}], tokenize=False, add_generation_prompt=True,
    enable_thinking=False), add_special_tokens=False), device=dev)
PLEN = PIDS.numel()
ACT = torch.from_numpy(np.load("/vol/data/holdout_fresh.npy").astype("float32"))[:NACT]

model = PeftModel.from_pretrained(base, os.path.join(RUN, CKPTS[0]), adapter_name="m").eval()
model.set_adapter("m")
TRAIN = [p for p in model.parameters() if p.requires_grad]

# ---- generate the frozen batch ONCE with the earliest checkpoint ----
with torch.no_grad():
    v = ACT.repeat_interleave(GROUP, 0).to(dev).float()
    HOOK["vec"] = v
    try:
        g = model.generate(input_ids=PIDS.unsqueeze(0).expand(v.shape[0], -1).contiguous(),
                           attention_mask=torch.ones(v.shape[0], PLEN, device=dev, dtype=torch.long),
                           do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                           max_new_tokens=39, pad_token_id=248046)
    finally:
        HOOK["vec"] = None
rolls = [[t for t in row if t != 248046] for row in g[:, PLEN:].tolist()]
# fixed advantages: standardise a synthetic-but-fixed reward so the DATA is identical across ckpts
rng = np.random.default_rng(0)
R = rng.normal(size=(NACT, GROUP)).astype("float32")
adv = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-6)
advf = torch.tensor(adv.reshape(-1), device=dev, dtype=torch.float32)
print("[replay] frozen batch: %d activations x %d rollouts, mean len %.1f"
      % (NACT, GROUP, float(np.mean([len(r) for r in rolls]))), flush=True)

def grad_norm_on_frozen(micro=8):
    for p in TRAIN:
        p.grad = None
    tot_lp, tot_n = 0.0, 0
    for a in range(0, len(rolls), micro):
        rs = rolls[a:a+micro]; ad = advf[a:a+micro]
        T = PLEN + max(len(r) for r in rs)
        ids = torch.full((len(rs), T), 248046, device=dev, dtype=torch.long)
        msk = torch.zeros((len(rs), T), device=dev, dtype=torch.bool)
        for j, r in enumerate(rs):
            ids[j, :PLEN] = PIDS
            ids[j, PLEN:PLEN+len(r)] = torch.tensor(r, device=dev)
            msk[j, PLEN:PLEN+len(r)] = True
        HOOK["vec"] = ACT.repeat_interleave(GROUP, 0)[a:a+micro].to(dev).float()
        tgt, m = ids[:, 1:], msk[:, 1:]
        lg = model(input_ids=ids).logits[:, :-1]
        lp = -F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                              reduction="none").view(tgt.shape)
        HOOK["vec"] = None
        per = -(ad.unsqueeze(1) * lp)
        loss = ((per * m).sum(1) / m.sum(1).clamp(min=1)).sum() / len(rolls)
        loss.backward()
        tot_lp += float((lp.detach() * m).sum()); tot_n += int(m.sum())
    gn = float(torch.norm(torch.stack([p.grad.norm() for p in TRAIN if p.grad is not None])))
    per_mod = {}
    for n_, p in model.named_parameters():
        if p.grad is None: continue
        key = "other"
        for k in ("o_proj", "down_proj", "q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"):
            if k in n_: key = k; break
        per_mod[key] = per_mod.get(key, 0.0) + float(p.grad.norm()) ** 2
    return gn, tot_lp / max(tot_n, 1), {k: v ** 0.5 for k, v in per_mod.items()}

out = []
for ck in CKPTS:
    path = os.path.join(RUN, ck)
    if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
        print("[replay] skip missing %s" % ck, flush=True); continue
    model.load_adapter(path, adapter_name="m", is_trainable=True)
    model.set_adapter("m")
    gn, lp, pm = grad_norm_on_frozen()
    out.append({"ckpt": ck, "grad_norm_frozen_batch": round(gn, 4),
                "mean_rollout_logp": round(lp, 4),
                "per_module": {k: round(v, 4) for k, v in sorted(pm.items())}})
    print("[replay] %-14s grad_norm %8.4f   mean_logp %8.4f   %s"
          % (ck, gn, lp, {k: round(v, 3) for k, v in sorted(pm.items())}), flush=True)
if len(out) > 1:
    f = out[-1]["grad_norm_frozen_batch"] / max(out[0]["grad_norm_frozen_batch"], 1e-9)
    print("\n[replay] on a FROZEN batch the norm changed %.2fx from %s to %s" % (f, out[0]["ckpt"], out[-1]["ckpt"]), flush=True)
    print("[replay] VERDICT: %s" % ("growth is IN THE MODEL (hypothesis B)" if f > 2.0
          else "growth is IN THE ROLLOUT DISTRIBUTION (hypothesis A)"), flush=True)
open("/vol/results_replay.json", "w").write(json.dumps(out, indent=1))
print("REPLAY_DONE", flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=5400,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def replay(run_dir: str = "/vol/ckpts/rl_modal_g16_t8_m39", ckpts: str = "", nact: int = 16):
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")
    if not ckpts:
        have = sorted(d for d in os.listdir(run_dir) if re.fullmatch(r"iter_\d+", d))
        pick = [have[0]] + ([have[len(have)//2]] if len(have) > 2 else []) + [have[-1]]
        ckpts = ",".join(dict.fromkeys(pick))
    open("/root/worker.py", "w").write(WORKER)
    env = dict(os.environ, REPLAY_RUN=run_dir, REPLAY_CKPTS=ckpts, REPLAY_NACT=str(nact))
    print("[replay] run=%s ckpts=%s" % (run_dir, ckpts), flush=True)
    p = subprocess.run(["python", "/root/worker.py"], env=env)
    VOL.commit()
    return p.returncode
