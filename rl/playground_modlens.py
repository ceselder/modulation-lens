"""Interactive modulation-lens playground: paste text -> read its L42 activation -> 4-bullet readout.

  modal deploy rl/playground_modlens.py     (then open the printed URL)

Serves BOTH the RL checkpoint and the SFT warm start on the same activation so the difference is
visible. Injection is REPLACE mode (h'_p = v, raw direction AND magnitude) because that is what the
lens was trained with; karvonen-style add loses 34% of the conditioning delta.
"""
import os, modal

app = modal.App("celeste-modlens-playground")
vol = modal.Volume.from_name("celeste-modlens-vol")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "transformers==5.15.0", "peft", "accelerate", "numpy",
                      "safetensors", "flash-linear-attention", "einops", "fastapi[standard]")
         .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir("src", "/root/src")
         .add_local_file("rl/playground_modlens.py", "/root/pg.py"))

BASE = "Qwen/Qwen3.6-27B"
LENSES = {
    "rl":  "/vol/ckpts_modlens_v3/final",     # delta 0.5322 (best measured)
    "sft": "/vol/av_sft_4b/final",            # delta 0.4768 (warm start)
}
PROMPT_FILE = "/vol/av_sft_4b/prompt.txt"
READ_LAYER = 42


@app.cls(image=image, volumes={"/vol": vol}, gpu="B200", timeout=3600,
         scaledown_window=900, max_containers=1)
class Lens:
    @modal.enter()
    def load(self):
        import sys, torch
        sys.path.insert(0, "/root"); sys.path.insert(0, "/root/src")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import inv_core as C
        self.torch, self.C = torch, C
        self.tok = AutoTokenizer.from_pretrained(BASE)
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
        self.model = PeftModel.from_pretrained(m, LENSES["rl"], adapter_name="rl").eval()
        self.model.load_adapter(LENSES["sft"], adapter_name="sft")
        self.inner = m.model
        self.INJ, self.LEFT, self.RIGHT = C.marker_ids(self.tok)
        self.HOOK = {"ids": None, "vec": None, "read": None}

        def stash(mod, args, kwargs):
            self.HOOK["ids"] = kwargs.get("input_ids", args[0] if args else None)

        def inject(mod, a, out):
            resid = out[0] if isinstance(out, tuple) else out
            ids, vec = self.HOOK["ids"], self.HOOK["vec"]
            if vec is None or ids is None or tuple(ids.shape) != tuple(resid.shape[:-1]):
                return out
            if not bool((ids == self.INJ).any()):
                return out
            new = C.inject_at_marker(ids, resid, vec, self.INJ, self.LEFT, self.RIGHT, "replace")
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

        def capture(mod, a, out):
            h = out[0] if isinstance(out, tuple) else out
            self.HOOK["read"] = h.detach().float()
            return out

        self.inner.register_forward_pre_hook(stash, with_kwargs=True)
        self.inner.layers[1].register_forward_hook(inject)
        self.inner.layers[READ_LAYER].register_forward_hook(capture)

        job = open(PROMPT_FILE).read()
        ptxt = self.tok.apply_chat_template([{"role": "user", "content": job}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        self.PIDS = torch.tensor(self.tok.encode(ptxt, add_special_tokens=False), device="cuda")
        at = (self.PIDS == self.INJ).nonzero().flatten()
        assert at.numel() == 1, "prompt needs exactly one marker"
        print("[ready] prompt %d tok, marker at %d" % (self.PIDS.shape[0], int(at[0])), flush=True)

    def _activation(self, text, pos=-1):
        """L42 residual at `pos` of `text` (default: last token). The lens reads what the model has
        just read, so pos=-1 means 'the state after consuming all of text'."""
        torch = self.torch
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        n = ids["input_ids"].shape[1]
        if n == 0:
            raise ValueError("empty text")
        p = (n + pos) if pos < 0 else min(pos, n - 1)
        self.HOOK["read"] = None
        with torch.no_grad():
            self.model(**ids)
        h = self.HOOK["read"][0, p].clone()
        self.HOOK["read"] = None
        return h, n, p

    @modal.method()
    def tokens(self, text: str):
        """Token strings for the picker. The lens reads ONE position's residual, so which token you
        pick is the whole experiment -- reading mid-sentence gives a different state than the end."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return {"tokens": [self.tok.decode([i]) for i in ids], "n": len(ids)}

    @modal.method()
    def read(self, text: str, pos: int = -1, max_new: int = 96, which: str = "both"):
        torch = self.torch
        h, n, p = self._activation(text, pos)
        out = {"n_tokens": n, "read_pos": p, "act_norm": float(h.norm()),
               "read_token": self.tok.decode(
                   self.tok(text, add_special_tokens=False)["input_ids"][max(0, p - 6):p + 1]),
               "readouts": {}}
        names = ["rl", "sft"] if which == "both" else [which]
        for name in names:
            self.model.set_adapter(name)
            self.HOOK["vec"] = h.unsqueeze(0)
            try:
                with torch.no_grad():
                    g = self.model.generate(
                        input_ids=self.PIDS.unsqueeze(0),
                        attention_mask=torch.ones(1, self.PIDS.shape[0], device="cuda", dtype=torch.long),
                        max_new_tokens=max_new, do_sample=False,
                        pad_token_id=self.tok.eos_token_id)
            finally:
                self.HOOK["vec"] = None
            out["readouts"][name] = self.tok.decode(
                g[0, self.PIDS.shape[0]:], skip_special_tokens=True).strip()
        return out


PAGE = """<!doctype html><meta charset=utf-8><title>modulation lens</title>
<style>
body{background:#faf7f2;color:#1a1a1a;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
     max-width:880px;margin:40px auto;padding:0 20px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#6b6b6b;font-size:13px;margin-bottom:22px}
textarea{width:100%;height:100px;padding:11px;border:1px solid #d8d0c4;border-radius:7px;
     font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#fff;resize:vertical}
button{padding:9px 20px;border:0;border-radius:7px;background:#c15f3c;color:#fff;
     font-size:14px;font-weight:600;cursor:pointer}button:disabled{opacity:.5}
button.ghost{background:#fff;color:#c15f3c;border:1px solid #e0cfc4}
.row{display:flex;gap:12px;align-items:center;margin-top:12px;font-size:13px;color:#555;flex-wrap:wrap}
select{padding:7px 9px;border:1px solid #d8d0c4;border-radius:6px;background:#fff;
     font:13px/1.4 ui-monospace,Menlo,monospace;max-width:420px}
input[type=number]{width:74px;padding:6px;border:1px solid #d8d0c4;border-radius:5px}
.card{margin-top:18px;padding:14px 16px;border:1px solid #e3dbcd;border-radius:8px;background:#fff}
.card h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#8a7f6d}
pre{margin:0;white-space:pre-wrap;font:14px/1.6 ui-monospace,Menlo,monospace}
.meta{color:#8a7f6d;font-size:12px;margin-top:14px}
.ex{color:#c15f3c;cursor:pointer;text-decoration:underline;font-size:13px;margin-right:10px}
</style>
<h1>modulation lens &mdash; 4-bullet readout</h1>
<div class=sub>Type text, pick which token's L42 state to read, and the lens describes that state in
four bullets. RL checkpoint vs the SFT warm start, on the same activation.</div>
<textarea id=t oninput="dirty()" placeholder="Almost all of my pieces are handmade,"></textarea>
<div class=row>
  <button class=ghost onclick=tokenize()>list tokens &#8595;</button>
  <select id=sel onchange="document.getElementById('p').value=this.value">
    <option value=-1>-1 &mdash; last token (default)</option>
  </select>
  <span>or index <input id=p type=number value=-1></span>
  <span>max tok <input id=m type=number value=96></span>
</div>
<div class=row><span>try:</span>
  <span class=ex onclick="setex(this)">Almost all of my pieces are handmade,</span>
  <span class=ex onclick="setex(this)">The defendant argued that the evidence had been obtained without a warrant,</span>
  <span class=ex onclick="setex(this)">I've been feeling like nobody at work actually listens to me anymore,</span>
</div>
<div class=row><button id=b onclick=go()>read the activation</button></div>
<div id=out></div>
<script>
function setex(e){document.getElementById('t').value=e.textContent; dirty(); tokenize()}
function dirty(){const s=document.getElementById('sel');
  if(s.dataset.for!==document.getElementById('t').value){
    s.innerHTML='<option value=-1>-1 &mdash; last token (default)</option>'; delete s.dataset.for}}
async function tokenize(){
  const txt=document.getElementById('t').value, s=document.getElementById('sel');
  if(!txt.trim())return;
  s.innerHTML='<option>tokenizing...</option>';
  const r=await fetch('tokens',{method:'POST',headers:{'Content-Type':'application/json'},
                                body:JSON.stringify({text:txt})});
  const j=await r.json();
  if(j.error){s.innerHTML='<option value=-1>-1 &mdash; last token</option>';return}
  let h='<option value=-1>-1 &mdash; last token ('+j.n+' tokens)</option>';
  j.tokens.forEach((tk,i)=>{
    const shown=tk.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])).replace(/ /g,'\u00b7');
    h+='<option value='+i+'>'+i+' \u2014 "'+shown+'"</option>';});
  s.innerHTML=h; s.dataset.for=txt;
}
async function go(){
  const b=document.getElementById('b'), out=document.getElementById('out');
  b.disabled=true; b.textContent='reading (first call loads 27B, ~5 min)...'; out.innerHTML='';
  try{
    const r=await fetch('read',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:document.getElementById('t').value,
                           pos:+document.getElementById('p').value,
                           max_new:+document.getElementById('m').value})});
    const j=await r.json();
    if(j.error){out.innerHTML='<div class=card><pre>'+j.error+'</pre></div>'}
    else{
      let h='';
      for(const k of Object.keys(j.readouts)){
        h+='<div class=card><h3>'+(k==='rl'?'RL checkpoint &middot; holdout delta 0.5322'
                                          :'SFT warm start &middot; holdout delta 0.4768')
          +'</h3><pre>'+j.readouts[k].replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))+'</pre></div>';
      }
      h+='<div class=meta>read L42 at position '+j.read_pos+' of '+j.n_tokens
        +' &middot; &#8214;h&#8214; '+j.act_norm.toFixed(1)
        +' &middot; the model had just read: <b>'
        +j.read_token.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))+'</b></div>';
      out.innerHTML=h;
    }
  }catch(e){out.innerHTML='<div class=card><pre>'+e+'</pre></div>'}
  b.disabled=false; b.textContent='read the activation';
}
</script>"""


@app.function(image=image, volumes={"/vol": vol})
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    api = FastAPI()

    @api.get("/")
    def index():
        return HTMLResponse(PAGE)

    @api.post("/tokens")
    async def tokens(req: Request):
        b = await req.json()
        try:
            return JSONResponse(Lens().tokens.remote(text=b.get("text") or ""))
        except Exception as e:
            return JSONResponse({"error": "%s: %s" % (type(e).__name__, e)})

    @api.post("/read")
    async def read(req: Request):
        b = await req.json()
        try:
            return JSONResponse(Lens().read.remote(
                text=b.get("text") or "", pos=int(b.get("pos", -1)),
                max_new=int(b.get("max_new", 96))))
        except Exception as e:
            return JSONResponse({"error": "%s: %s" % (type(e).__name__, e)})

    return api
