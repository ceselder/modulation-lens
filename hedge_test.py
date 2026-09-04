"""Is the hedging ("random", "a snippet from a transcript", "something") a DECODING artefact?

3.0% of emitted lines contained "random" and 20.4% retreated to describing the medium, in rollouts
sampled at temperature 1.0. A greedy single-line read of a different checkpoint showed 1 hedge in 64.
Those two differ in BOTH temperature and format, so the comparison was confounded. This holds the
checkpoint and the format fixed and varies only temperature.

If hedging is temperature-driven the fix is decoding-side -- low temperature or best-of-N at readout
-- which costs nothing and needs no retraining. If it persists under greedy decoding, it is baked
into the policy and needs a training-side answer.
"""
import os
import subprocess

import modal

app = modal.App("celeste-modlens-hedge")
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
CKPT=os.environ["H_CKPT"]; NACT=int(os.environ.get("H_NACT","64")); dev="cuda"
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
inner=base.model
INJ,LEFT,RIGHT=C.marker_ids(tok)
HOOK={"vec":None,"ids":None}
inner.register_forward_pre_hook(lambda m,a,kw: HOOK.__setitem__("ids",
    kw.get("input_ids") if kw.get("input_ids") is not None else (a[0] if a else None)), with_kwargs=True)
def _inj(m,a,out):
    r=out[0] if isinstance(out,tuple) else out
    ids,vec=HOOK["ids"],HOOK["vec"]
    if vec is None or ids is None or tuple(ids.shape)!=tuple(r.shape[:-1]): return out
    if not bool((ids==INJ).any()): return out
    n=C.inject_at_marker(ids,r,vec,INJ,LEFT,RIGHT,mode="karvonen")
    return (n,)+tuple(out[1:]) if isinstance(out,tuple) else n
inner.layers[1].register_forward_hook(_inj)
model=PeftModel.from_pretrained(base,CKPT,adapter_name="m").eval(); model.set_adapter("m")
# Reproduce inv_train's prompt EXACTLY. With --bullets 1 and --grid-templates recovered none of
# the conditional patches fire, so this is the verbatim base string; the token-count assert below
# is the fingerprint the RL run printed ("[prompt] 186 tokens"), so a silent drift becomes a crash.
JOB = ("You are shown an internal activation vector captured from a language model at a single "
       "position while it was reading some text. The vector is enclosed in <concept> tags.\n\n"
       "<concept>%s</concept>\n\n"
       "Your job: write the short phrase that this state is holding in mind.\n\n"
       "How it is judged. Your phrase is placed into a prompt of the form\n"
       '  Focus on the following idea: "<your phrase>" while writing the following phrase: '
       '"<a fixed unrelated sentence>"\n'
       "The model then writes that fixed sentence, and we read its internal state while it does "
       "so. You score well when that state matches the state you were given.\n\n"
       "So write what the model should be THINKING ABOUT -- not a description of a vector, and not "
       "a comment on the task. Natural, fluent English. At most %d tokens. Output only the phrase."
       % (C.INJ_CHAR, 16))
PIDS=torch.tensor(tok.encode(tok.apply_chat_template([{"role":"user","content":JOB}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False),
    add_special_tokens=False), device=dev)
PLEN=PIDS.numel()
assert PLEN==186, "prompt drifted: %d tokens, RL run had 186" % PLEN
assert int((PIDS==INJ).sum())==1, "marker count wrong"
print("[prompt] %d tokens, marker ok" % PLEN, flush=True)
ACT=torch.from_numpy(np.load("/vol/data/holdout_fresh.npy").astype("float32"))[:NACT]
HEDGE=re.compile(r"\b(random|maybe|perhaps|possibly|snippet|transcript|corpus|something|unclear|vague|kinda|sort of)\b", re.I)
_BUL=re.compile(r"^\s*(?:[*•\-–]|\d+[.)])\s+")
def gen(temp):
    with torch.no_grad():
        HOOK["vec"]=ACT.to(dev).float()
        try:
            kw=dict(do_sample=False) if temp<=0 else dict(do_sample=True, temperature=temp, top_p=1.0, top_k=0)
            g=model.generate(input_ids=PIDS.unsqueeze(0).expand(NACT,-1).contiguous(),
                             attention_mask=torch.ones(NACT,PLEN,device=dev,dtype=torch.long),
                             max_new_tokens=16, pad_token_id=248046, **kw)
        finally:
            HOOK["vec"]=None
    out=[]
    for t in tok.batch_decode(g[:,PLEN:], skip_special_tokens=True):
        first=next((l for l in t.splitlines() if l.strip()), t)
        out.append(_BUL.sub("",first).strip())
    return out
res={}
for label,temp in (("greedy",0.0),("temp 0.7",0.7),("temp 1.0",1.0)):
    ps=gen(temp)
    h=[p for p in ps if HEDGE.search(p)]
    res[label]={"n":len(ps),"hedged":len(h),"rate":len(h)/max(len(ps),1),
                "mean_words":float(np.mean([len(p.split()) for p in ps])),
                "examples_hedged":h[:4],"examples_clean":[p for p in ps if not HEDGE.search(p)][:4]}
    print("[hedge] %-9s %2d/%2d = %5.1f%% hedged   mean %.1f words" %
          (label,len(h),len(ps),100*len(h)/max(len(ps),1),res[label]["mean_words"]), flush=True)
    for e in h[:3]: print("           hedged: %s" % e[:70], flush=True)
    for e in res[label]["examples_clean"][:2]: print("           clean : %s" % e[:70], flush=True)
json.dump(res, open("/vol/results_hedge.json","w"), indent=1)
print("HEDGE_DONE", flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=3600,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def hedge(ckpt: str = "/vol/ckpts/sft_oneline/final", nact: int = 64):
    os.makedirs("/workspace", exist_ok=True)
    if not os.path.exists("/workspace/.hf_home"):
        os.symlink("/vol/.hf_home", "/workspace/.hf_home")
    open("/root/worker.py", "w").write(WORKER)
    p = subprocess.run(["python", "/root/worker.py"],
                       env=dict(os.environ, H_CKPT=ckpt, H_NACT=str(nact)))
    VOL.commit()
    return p.returncode
