"""Modulation-lens readout at every PUNCTUATION position of the blogpost, keyed to the j-lens eval.

The existing jlens-blogpost-viewer has single-token OMP readouts for all 3570 positions (577 of them
punctuation). At punctuation the single-token readout is where the j-lens struggles most -- e.g. at
"She's being vulnerable on main again." it returns [' Twitter', 'oo', ' woman', '-vesm']: real signal
(on main = Twitter, woman) buried in junk, because one token cannot say "she is posting something
personal publicly". A 4-bullet phrase lens can. This produces that side of the comparison.

Positions are identified by (para, i) taken from omp_readout_all.json, so the output joins to the
existing viewer row-for-row.

  modal run rl/blogpost_modlens.py::main --positions /vol/data/blogpost_punct_positions.json
"""
import json, os, modal

app = modal.App("modlens-blogpost-readout")
vol = modal.Volume.from_name("celeste-modlens-vol")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "transformers==5.15.0", "peft", "accelerate", "numpy",
                      "safetensors", "flash-linear-attention", "einops")
         .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir("src", "/root/src")
         .add_local_file("rl/blogpost_modlens.py", "/root/b.py"))

BASE = "Qwen/Qwen3.6-27B"
LENSES = {"rl": "/vol/ckpts_modlens_v3/final", "sft": "/vol/av_sft_4b/final"}
PROMPT_FILE = "/vol/av_sft_4b/prompt.txt"
READ_LAYER = 42
OUT = "/vol/data/blogpost_modlens_readout.json"


@app.function(image=image, volumes={"/vol": vol}, gpu="B200", timeout=14400)
def run(positions_path: str, max_new: int = 96, batch: int = 32, limit: int = 0):
    import sys, torch
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/src")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import inv_core as C

    paras = json.load(open(positions_path))          # {"paragraphs": {para: text}, "positions": [...]}
    P = paras["positions"][: limit or None]
    PT = {int(k): v for k, v in paras["paragraphs"].items()}
    print("[in] %d positions across %d paragraphs" % (len(P), len(PT)), flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
    model = PeftModel.from_pretrained(m, LENSES["rl"], adapter_name="rl").eval()
    model.load_adapter(LENSES["sft"], adapter_name="sft")
    inner = m.model
    INJ, LEFT, RIGHT = C.marker_ids(tok)
    HOOK = {"ids": None, "vec": None, "read": None}

    def stash(mod, a, kw):
        HOOK["ids"] = kw.get("input_ids", a[0] if a else None)

    def inject(mod, a, out):
        resid = out[0] if isinstance(out, tuple) else out
        ids, vec = HOOK["ids"], HOOK["vec"]
        if vec is None or ids is None or tuple(ids.shape) != tuple(resid.shape[:-1]):
            return out
        if not bool((ids == INJ).any()):
            return out
        new = C.inject_at_marker(ids, resid, vec, INJ, LEFT, RIGHT, "replace")
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    def capture(mod, a, out):
        h = out[0] if isinstance(out, tuple) else out
        HOOK["read"] = h.detach().float()
        return out

    inner.register_forward_pre_hook(stash, with_kwargs=True)
    inner.layers[1].register_forward_hook(inject)
    inner.layers[READ_LAYER].register_forward_hook(capture)

    job = open(PROMPT_FILE).read()
    ptxt = tok.apply_chat_template([{"role": "user", "content": job}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    PIDS = torch.tensor(tok.encode(ptxt, add_special_tokens=False), device="cuda")
    assert (PIDS == INJ).sum().item() == 1, "prompt needs exactly one marker"

    # --- activations: one forward per paragraph, read every requested position in it ---
    want = {}
    for p in P:
        want.setdefault(int(p["para"]), []).append(int(p["i"]))
    acts, misaligned = {}, 0
    with torch.no_grad():
        for para, idxs in sorted(want.items()):
            ids = tok(PT[para], return_tensors="pt", add_special_tokens=False).to("cuda")
            n = ids["input_ids"].shape[1]
            HOOK["read"] = None
            model(**ids)
            h = HOOK["read"][0]
            for i in idxs:
                if i >= n:
                    misaligned += 1; continue
                acts[(para, i)] = h[i].clone()
            HOOK["read"] = None
    print("[act] %d activations (%d positions past paragraph end -> skipped)"
          % (len(acts), misaligned), flush=True)

    # --- readouts, greedy, both lenses ---
    keys = [k for k in ((int(p["para"]), int(p["i"])) for p in P) if k in acts]
    out_rows = {}
    for name in ("rl", "sft"):
        model.set_adapter(name)
        for s in range(0, len(keys), batch):
            ch = keys[s:s + batch]
            V = torch.stack([acts[k] for k in ch])
            HOOK["vec"] = V
            try:
                with torch.no_grad():
                    g = model.generate(
                        input_ids=PIDS.unsqueeze(0).expand(len(ch), -1).contiguous(),
                        attention_mask=torch.ones(len(ch), PIDS.shape[0], device="cuda", dtype=torch.long),
                        max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            finally:
                HOOK["vec"] = None
            for k, t in zip(ch, tok.batch_decode(g[:, PIDS.shape[0]:], skip_special_tokens=True)):
                out_rows.setdefault("%d_%d" % k, {})[name] = t.strip()
            if s % (batch * 20) == 0:
                print("  [%s] %d/%d" % (name, s + len(ch), len(keys)), flush=True)

    meta = {int(p["para"]) * 100000 + int(p["i"]): p for p in P}
    res = {"config": {"lens_rl": LENSES["rl"], "lens_sft": LENSES["sft"], "layer": READ_LAYER,
                      "inject": "replace", "greedy": True, "max_new": max_new,
                      "n_positions": len(keys), "skipped_misaligned": misaligned},
           "rows": [{"para": int(p["para"]), "i": int(p["i"]), "tok": p.get("tok"),
                     "ctx": p.get("ctx"),
                     "rl": out_rows.get("%d_%d" % (int(p["para"]), int(p["i"])), {}).get("rl"),
                     "sft": out_rows.get("%d_%d" % (int(p["para"]), int(p["i"])), {}).get("sft")}
                    for p in P if (int(p["para"]), int(p["i"])) in acts]}
    os.makedirs("/vol/data", exist_ok=True)
    json.dump(res, open(OUT, "w"))
    vol.commit()
    print("[done] %d rows -> %s" % (len(res["rows"]), OUT), flush=True)
    for r in res["rows"][:4]:
        print("  %r ctx %r\n    rl : %s" % (r["tok"], (r["ctx"] or "")[-60:],
                                            (r["rl"] or "").replace("\n", " | ")), flush=True)
    return {"n": len(res["rows"]), "out": OUT}


@app.local_entrypoint()
def main(positions: str = "/vol/data/blogpost_punct_positions.json", limit: int = 0):
    print(run.remote(positions_path=positions, limit=limit))
