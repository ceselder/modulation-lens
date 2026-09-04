#!/usr/bin/env python3
"""
Resident inverter playground. Model stays loaded; each query is seconds.

Paste text -> click a token -> the inverter says what the hidden state at that token is holding,
with each candidate scored by the modulation reward. Optionally refine with Sonnet.

  CUDA_VISIBLE_DEVICES=1 PORT=8813 python inv_server.py
"""
import json, os, re, sys, threading, time
import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inv_core as C

PORT = int(os.environ.get("PORT", "8813"))
MAXTOK = int(os.environ.get("MAXTOK", "3072"))   # paste a whole document
NOJAC = int(os.environ.get("NOJAC", "0"))   # 1 = raw layer-42 space, J = identity
LAYER = 42
CKPTS = {"sft (baseline: copies the text)": "/workspace/inv/ckpts/sft/final"}
_ARMS = ((("rl_noJ", "noJ", "no-jacobian"),) if NOJAC else
          (("rl_v2_plain", "plain", "plain cosine"), ("rl_v2_whit", "whit", "whitened")))
for _arm, _tag, _note in _ARMS:
    _d = "/workspace/inv/ckpts/%s" % _arm
    if not os.path.isdir(_d):
        continue
    _its = sorted(int(x.split("_")[1]) for x in os.listdir(_d) if x.startswith("iter_"))
    for _n in _its:
        if _n % 100 == 0 or _n == max(_its):          # keep the dropdown readable
            CKPTS["%s-%d%s" % (_tag, _n, " (last)" if _n == max(_its) else "")] = \
                "%s/iter_%06d" % (_d, _n)
    if os.path.isdir(_d + "/final"):
        CKPTS["%s-final" % _tag] = _d + "/final"
dev = "cuda"
LOCK = threading.Lock()

print("[s] loading base...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
INJ, LEFT, RIGHT = C.marker_ids(tok)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
J = C.load_jlens(LAYER, dev)
if NOJAC:
    J = torch.eye(J.shape[0], device=dev, dtype=J.dtype)   # raw L42 space, no Jacobian
    print("[s] NO-JACOBIAN mode: scoring in raw layer-42 space", flush=True)
MU, Wm = C.load_whitener("/workspace/inv/data/meansub/natural_whitener_jspace.npz", "0.1", dev)
WHITEN = int(os.environ.get("WHITEN", "0"))   # 0 = plain cosine (what the main arm trained on)


def _w(x):
    """Apply the whitener only if this server is running in whitened mode."""
    return x @ Wm.T if WHITEN else x


GRID = C.Grid(tok, C.TEMPLATES_RECOVERED, C.CARRIERS_RECOVERED, LAYER, J, dev)
L42 = {}
base.model.layers[LAYER].register_forward_hook(
    lambda m, i, o: L42.__setitem__("h", o[0] if isinstance(o, tuple) else o))
CPRE, CPOST = C.chat_wrap_ids(tok)

import pyarrow.parquet as pq
acc = []
for b in pq.ParquetFile("/workspace/inv/data/prose_L42.parquet").iter_batches(
        batch_size=4096, columns=["activation_vector"]):
    acc.append(np.array(b.to_pydict()["activation_vector"], dtype="float32"))
    if sum(len(x) for x in acc) >= 20000:
        break
AMU = torch.from_numpy(np.concatenate(acc)[:20000]).mean(0).to(dev) @ J.T
_SIG = GRID.sig()[:10] + ("_noJ" if NOJAC else "")
_PMU_P = os.environ.get("PMU_PATH", ("/workspace/inv/ckpts/rl_noJ/pmu_%s.npy" % _SIG) if NOJAC else
    ("/workspace/inv/ckpts/rl_v2_plain/pmu_%s.npy" % _SIG))
assert os.path.exists(_PMU_P), (
    "no PMU for grid %s at %s -- the old ckpts/rl/pmu.npy is the GRID V1 centre and using it "
    "here would score every phrase against the wrong mean." % (_SIG, _PMU_P))
PMU = torch.tensor(np.load(_PMU_P), device=dev)
print("[s] |PMU| %.2f |AMU| %.2f | grid %dx%d" % (float(PMU.norm()), float(AMU.norm()),
      GRID.n_tpl, GRID.n_car), flush=True)

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
       "a comment on the task. Natural, fluent English. At most 16 tokens. Output only the phrase.")
PIDS = torch.tensor(tok.encode(tok.apply_chat_template(
    [{"role": "user", "content": JOB % C.INJ_CHAR}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False), add_special_tokens=False), device=dev)
PLEN = PIDS.numel()

ADAPTERS = {}
for i, (name, path) in enumerate(CKPTS.items()):
    if not os.path.exists(path):
        print("[s] missing %s" % path, flush=True); continue
    an = "a%d" % i
    if not ADAPTERS:
        PM = PeftModel.from_pretrained(base, path, adapter_name=an)
    else:
        PM.load_adapter(path, adapter_name=an)
    ADAPTERS[name] = an
    print("[s] loaded %s -> %s" % (name, an), flush=True)
inner = PM.base_model.model.model
HK = {"vec": None, "ids": None}
inner.register_forward_pre_hook(
    lambda m, a, kw: HK.__setitem__("ids", kw.get("input_ids", a[0] if a else None)),
    with_kwargs=True)


def _inj(m, a, o):
    r = o[0] if isinstance(o, tuple) else o
    if HK["vec"] is None or HK["ids"] is None:
        return o
    if tuple(HK["ids"].shape) != tuple(r.shape[:-1]) or not bool((HK["ids"] == INJ).any()):
        return o
    n = C.inject_at_marker(HK["ids"], r, HK["vec"], INJ, LEFT, RIGHT)
    return (n,) + tuple(o[1:]) if isinstance(o, tuple) else n


inner.layers[1].register_forward_hook(_inj)


@torch.no_grad()
def tokens_of(text):
    return tok(text, add_special_tokens=False).input_ids[:MAXTOK]


@torch.no_grad()
def state_at(text, pos, role="user"):
    """Read layer 42 over the passage, with the passage on either conversational turn.

    role="user"      -> <|im_start|>user\n TEXT <|im_end|> ... assistant <think></think>
                        i.e. text the model was GIVEN.
    role="assistant" -> empty user turn, then assistant <think></think> TEXT
                        i.e. text the model is WRITING. Same token lists, text moved after the
                        assistant header, so no re-tokenization of the passage either way.

    In both cases the passage is inserted whole and the read is sliced at its own offset, so
    position i of the passage is position i of the readout.
    """
    ids = tokens_of(text)
    pos = max(0, min(pos if pos >= 0 else len(ids) - 1, len(ids) - 1))
    if role == "assistant":
        seq, off = CPRE + CPOST + ids, len(CPRE) + len(CPOST)
    else:
        seq, off = CPRE + ids + CPOST, len(CPRE)
    with PM.disable_adapter():
        base(input_ids=torch.tensor([seq], device=dev))
    H = L42["h"].float()[0][off:off + len(ids)]
    return H[pos].clone(), ids, pos


@torch.no_grad()
def propose(vec, adapter, n, temp, max_new=16):
    PM.set_adapter(adapter)
    HK["vec"] = vec.unsqueeze(0).expand(n, -1).contiguous().float()
    try:
        g = PM.generate(input_ids=PIDS.unsqueeze(0).expand(n, -1).contiguous(),
                        attention_mask=torch.ones(n, PLEN, device=dev, dtype=torch.long),
                        max_new_tokens=max_new, do_sample=True, temperature=temp,
                        top_p=1.0, top_k=0,
                        pad_token_id=tok.eos_token_id)
    finally:
        HK["vec"] = None
    return [t.strip() for t in tok.batch_decode(g[:, PLEN:], skip_special_tokens=True) if t.strip()]


@torch.no_grad()
def score(strings, vec, ncar, max_tok=16):
    t = _w((vec @ J.T) - AMU)
    t = t / t.norm().clamp(min=1e-8)
    acc = {s: torch.zeros(J.shape[0], device=dev) for s in strings}
    with PM.disable_adapter():
        for c in range(max(1, min(ncar, GRID.n_car))):
            v = GRID.read(base, strings, L42, carrier=c, max_tok=max_tok)
            for s in strings:
                acc[s] += v[s]
    out = {}
    for s in strings:
        a = _w(acc[s] / max(1, min(ncar, GRID.n_car)) - PMU)
        out[s] = float((a @ t) / a.norm().clamp(min=1e-8))
    return out


W_E = base.get_input_embeddings().weight
VOCAB = W_E.shape[0]
_pat = __import__("re").compile(r"^ ?[A-Za-z][A-Za-z\'-]{1,}$")
VMASK = torch.zeros(VOCAB, dtype=torch.bool)
for _t, _d in enumerate(tok.batch_decode([[i] for i in range(VOCAB)])):
    if _d and _d.isascii() and _pat.match(_d):
        VMASK[_t] = True
VMASK = VMASK.to(dev)
print("[s] %d word-ish tokens allowed for search" % int(VMASK.sum()), flush=True)
JOBS, JLOCK = {}, threading.Lock()


def _score_from_vecs(acc, ncell, tw):
    a = _w(acc / ncell - PMU)
    return (a @ tw) / a.norm(dim=-1).clamp(min=1e-8)


def gcg_grad(opt_ids, tw, cells):
    """d(score)/d(one-hot) at opt_ids, summed over the given cells. opt_ids [1, L]."""
    oh = torch.zeros(1, opt_ids.shape[1], VOCAB, dtype=W_E.dtype, device=dev)
    oh.scatter_(2, opt_ids.unsqueeze(2), 1.0)
    oh.requires_grad_(True)
    mid = oh @ W_E
    tot = None
    for S in cells:
        pre = W_E[torch.tensor(S["pre"], device=dev)].unsqueeze(0)
        post = W_E[torch.tensor(S["post"], device=dev)].unsqueeze(0)
        base(inputs_embeds=torch.cat([pre, mid, post], dim=1))
        v = L42["h"][:, -S["ncar"]:, :].float().mean(1) @ J.T
        tot = v if tot is None else tot + v
    sc = _score_from_vecs(tot, len(cells), tw)
    (-sc.sum()).backward()
    g = -oh.grad[0]
    return g.masked_fill(~VMASK.unsqueeze(0), float("-inf"))


@torch.no_grad()
def score_ids(cand, tw, cells):
    """cand [B, L] -> [B] score, batched (all rows same length, so no padding in the slot)."""
    tot = None
    for S in cells:
        pre = torch.tensor(S["pre"], device=dev).unsqueeze(0).expand(cand.shape[0], -1)
        post = torch.tensor(S["post"], device=dev).unsqueeze(0).expand(cand.shape[0], -1)
        base(input_ids=torch.cat([pre, cand, post], dim=1))
        v = L42["h"].float()[:, -S["ncar"]:, :].mean(1) @ J.T
        tot = v if tot is None else tot + v
    return _score_from_vecs(tot, len(cells), tw)


def gcg_run(job, text, pos, init, steps, n_opt, topk, batch, ncar, grad_cells,
            role="user"):
    try:
        with LOCK:
            vec, ids, pos = state_at(text, pos, role)
        tw = _w((vec @ J.T) - AMU)
        tw = tw / tw.norm().clamp(min=1e-8)
        gcells = GRID.cells[0][: max(1, grad_cells)]
        scells = [c for row in GRID.cells[: max(1, ncar)] for c in row]
        if init.strip():
            t = tok(init.strip(), add_special_tokens=False).input_ids
            cur = torch.tensor([(t * (n_opt // max(1, len(t)) + 1))[:n_opt]], device=dev)
        else:
            allow = VMASK.nonzero().flatten()
            cur = allow[torch.randint(0, len(allow), (1, n_opt), device=dev)]
        best_s, best_t = -9.0, ""
        for it in range(steps):
            with JLOCK:
                if JOBS[job].get("cancel"):
                    break
            with LOCK:
                g = gcg_grad(cur, tw, gcells)
                tk = g.topk(topk, dim=1).indices                       # [L, topk]
                ps = torch.randint(0, n_opt, (batch,), device=dev)
                pk = tk[ps, torch.randint(0, topk, (batch,), device=dev)]
                cand = cur.repeat(batch, 1)
                cand[torch.arange(batch, device=dev), ps] = pk
                cand = torch.cat([cur, cand], 0)
                sc = score_ids(cand, tw, scells)
                k = int(sc.argmax())
                cur = cand[k:k + 1].clone()
                cs = float(sc[k])
            txt = tok.decode(cur[0])
            if cs > best_s:
                best_s, best_t = cs, txt
            with JLOCK:
                JOBS[job].update(step=it + 1, cur=txt, cur_score=round(cs, 4),
                                 best=best_t, best_score=round(best_s, 4))
        with JLOCK:
            JOBS[job]["done"] = True
    except Exception as e:
        with JLOCK:
            JOBS[job].update(error="%s: %s" % (type(e).__name__, e), done=True)


HTML = """<!doctype html><meta charset=utf-8><title>inverter</title>
<style>
body{font:14px/1.5 ui-sans-serif,system-ui;margin:0;background:#faf8f4;color:#241f19}
main{max-width:1080px;margin:0 auto;padding:18px}
h1{font-size:19px;margin:0 0 2px} .sub{color:#6b6355;font-size:12.5px;margin:0 0 14px}
textarea{width:100%;height:84px;font:13px ui-monospace,Menlo,monospace;padding:8px;
 border:1px solid #d8d0c2;border-radius:7px;background:#fff}
.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:9px 0}
button{padding:6px 13px;border:1px solid #b8552f;background:#b8552f;color:#fff;border-radius:6px;
 cursor:pointer;font-size:13px} button.g{background:#fff;color:#8a4527}
select,input{padding:5px 7px;border:1px solid #d8d0c2;border-radius:6px;background:#fff;font-size:13px}
#toks{background:#fff;border:1px solid #e2dbcd;border-radius:7px;padding:9px;line-height:2.2;
 font:12.5px ui-monospace,Menlo,monospace;max-height:190px;overflow:auto}
.tk{padding:2px 3px;border-radius:3px;cursor:pointer;white-space:pre}
.tk:hover{background:#f0e6d6} .tk.on{background:#b8552f;color:#fff}
table{width:100%;border-collapse:collapse;margin-top:11px;background:#fff}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #eee6d8;font-size:13px}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:#6b6355}
td.s{font-variant-numeric:tabular-nums;width:5.5em;font-weight:600}
td.p{font:12.5px ui-monospace,Menlo,monospace}
.pill{font-size:11.5px;color:#6b6355} .warn{color:#8a4527}
</style>
<main>
<h1>activation &rarr; phrase</h1>
<p class="sub">Paste text, click a token, and the inverter says what the layer-42 state at that token
is holding. Each candidate is scored by the modulation reward. <span class="warn">rl350 is the
reward-hacked checkpoint &mdash; it will say &ldquo;not pizza related wording&rdquo; for
everything.</span></p>
<textarea id="t">do you want to make it or do you just want to be held?</textarea>
<div class="row">
 <button id="tk">tokenize</button>
 <select id="ck"></select>
 <label class="pill">samples <input id="n" type="number" value="12" min="1" max="48" style="width:56px"></label>
 <label class="pill">max tok <input id="mx" type="number" value="16" min="4" max="128" style="width:56px"></label>
 <label class="pill">turn <select id="rl"><option value="user">user (given)</option><option value="assistant">assistant (writing)</option></select></label>
 <label class="pill">temp <input id="tp" type="number" value="1.1" min="0.1" max="2" step="0.1" style="width:56px"></label>
 <label class="pill">carriers <input id="nc" type="number" value="3" min="1" max="6" style="width:52px"></label>
 <button id="go">read this token</button>
 <button class="g" id="ev">+ refine with Sonnet</button>
 <span class="pill" id="st"></span>
</div>
<div id="toks"></div>
<table id="out"><thead><tr><th>score</th><th>phrase</th></tr></thead><tbody></tbody></table>

<h1 style="margin-top:26px">evolve a prompt for the same token</h1>
<p class="sub">Gradient search (GCG) on the same target: it edits one token at a time, keeping
whatever raises the score. Seed it with the inverter's best phrase &mdash; that is usually a much
better starting point than random tokens. Search scores on 1 carrier for speed; the numbers are
therefore noisier than the readout table above.</p>
<div class="row">
 <button id="sgo">start search</button>
 <button class="g" id="sstop">stop</button>
 <button class="g" id="sseed">seed from best readout</button>
 <label class="pill">steps <input id="ss" type="number" value="80" min="1" max="4000" style="width:64px"></label>
 <label class="pill">slot <input id="sn" type="number" value="14" min="2" max="48" style="width:52px"></label>
 <label class="pill">top-k <input id="sk" type="number" value="256" min="8" max="512" step="8" style="width:62px"></label>
 <label class="pill">batch <input id="sb" type="number" value="48" min="4" max="192" style="width:56px"></label>
 <span class="pill" id="sst"></span>
</div>
<div class="row"><input id="si" placeholder="initial string (blank = random word tokens)" style="flex:1"></div>
<table id="sout"><thead><tr><th>score</th><th>string</th></tr></thead><tbody></tbody></table>
</main>
<script>
let POS=-1, IDS=[];
const el=i=>document.getElementById(i);
fetch('/meta').then(r=>r.json()).then(m=>{
  el('ck').innerHTML=m.ckpts.map(c=>`<option value="${c.id}">${c.label}</option>`).join('');
  el('st').textContent=m.ckpts.length+' checkpoints loaded';
}).catch(e=>{el('st').textContent='COULD NOT LOAD CHECKPOINT LIST: '+e;});
el('tk').onclick=async()=>{
  const r=await(await fetch('/tok',{method:'POST',body:JSON.stringify({text:el('t').value})})).json();
  IDS=r.toks; POS=IDS.length-1;
  el('toks').innerHTML=IDS.map((s,i)=>`<span class=tk data-i=${i}>${s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/ /g,'\\u00b7')}</span>`).join('');
  [...document.querySelectorAll('.tk')].forEach(n=>n.onclick=()=>{
    POS=+n.dataset.i; document.querySelectorAll('.tk').forEach(x=>x.classList.remove('on'));
    n.classList.add('on'); el('st').textContent='token '+POS+' = '+IDS[POS];});
  document.querySelectorAll('.tk')[POS].classList.add('on');
  el('st').textContent=IDS.length+' tokens; last selected';
};
async function run(refine){
  if(!IDS.length) await el('tk').onclick();
  el('st').textContent=refine?'proposing + refining…':'reading…';
  if(!el('ck').value){el('st').textContent='no checkpoint selected (list failed to load?)';return}
  const r=await(await fetch(refine?'/evolve':'/read',{method:'POST',body:JSON.stringify({
    text:el('t').value,pos:POS,ckpt:el('ck').value,n:+el('n').value,
    temp:+el('tp').value,ncar:+el('nc').value,max_new:+el('mx').value,role:el('rl').value})})).json();
  if(r.error){el('st').textContent=r.error;return}
  el('st').textContent='read with: '+(r.ckpt_used||'?')+'  @ token '+r.pos+'  ('+r.secs+'s)';
  el('out').querySelector('tbody').innerHTML=r.rows.map(x=>
    `<tr><td class=s>${x.score.toFixed(4)}</td><td class=p>${x.phrase.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</td></tr>`).join('');
  el('st').textContent=`token ${r.pos} = ${r.token}  ·  ${r.rows.length} candidates  ·  ${r.secs}s`;
}
el('go').onclick=()=>run(false); el('ev').onclick=()=>run(true);

let SJ=null,ST=null;
el('sseed').onclick=()=>{
  const r=el('out').querySelector('tbody tr');
  if(r){el('si').value=r.children[1].textContent; el('sst').textContent='seeded';}
  else el('sst').textContent='read a token first';
};
el('sgo').onclick=async()=>{
  if(ST){clearInterval(ST);ST=null}
  el('sout').querySelector('tbody').innerHTML='';
  el('sst').textContent='starting…';
  const r=await(await fetch('/search',{method:'POST',body:JSON.stringify({
    text:el('t').value,pos:POS,init:el('si').value,steps:+el('ss').value,
    n_opt:+el('sn').value,topk:+el('sk').value,batch:+el('sb').value,ncar:1,grad_cells:3})})).json();
  if(r.error){el('sst').textContent=r.error;return}
  SJ=r.job; const seen=new Set();
  ST=setInterval(async()=>{
    const s=await(await fetch('/gcg?job='+SJ)).json();
    if(s.error){el('sst').textContent=s.error;clearInterval(ST);ST=null;return}
    el('sst').textContent=`step ${s.step}/${s.total} · best ${(s.best_score||0).toFixed(4)}`;
    const key=(s.best_score||0)+'|'+(s.best||'');
    if(s.best && !seen.has(key)){seen.add(key);
      const tb=el('sout').querySelector('tbody');
      const tr=document.createElement('tr');
      tr.innerHTML=`<td class=s>${(s.best_score||0).toFixed(4)}</td><td class=p>${(s.best||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')}</td>`;
      tb.insertBefore(tr,tb.firstChild);}
    if(s.done){clearInterval(ST);ST=null;el('sst').textContent+=' · done'}
  },1500);
};
el('sstop').onclick=async()=>{if(SJ)await fetch('/gcg?job='+SJ+'&cancel=1')};

el('tk').onclick();
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _s(self, code, body, ct="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            return self._s(200, HTML, "text/html; charset=utf-8")
        if self.path.startswith("/meta"):
            return self._s(200, json.dumps(
                {"ckpts": [{"id": v, "label": k} for k, v in ADAPTERS.items()],
                 "maxtok": MAXTOK}))
        if self.path.startswith("/gcg"):
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(self.path).query)
            jid = (q.get("job") or [""])[0]
            with JLOCK:
                if q.get("cancel") and jid in JOBS:
                    JOBS[jid]["cancel"] = True
                st = dict(JOBS.get(jid, {"error": "no such job"}))
            return self._s(200, json.dumps(st))
        self._s(404, "{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            q = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._s(400, json.dumps({"error": "bad json"}))
        t0 = time.time()
        try:
            if self.path.startswith("/search"):
                jid = "g%d" % (len(JOBS) + 1)
                with JLOCK:
                    JOBS[jid] = {"step": 0, "total": int(q.get("steps", 60)), "done": False,
                                 "best": "", "best_score": 0.0}
                threading.Thread(target=gcg_run, daemon=True, kwargs=dict(
                    job=jid, text=q.get("text", ""), pos=int(q.get("pos", -1)),
                    init=str(q.get("init", "")), steps=max(1, min(int(q.get("steps", 60)), 4000)),
                    n_opt=max(2, min(int(q.get("n_opt", 14)), 48)),
                    topk=max(8, min(int(q.get("topk", 256)), 512)),
                    batch=max(4, min(int(q.get("batch", 48)), 192)),
                    ncar=max(1, min(int(q.get("ncar", 1)), GRID.n_car)),
                    grad_cells=max(1, min(int(q.get("grad_cells", 3)), GRID.n_tpl)))).start()
                return self._s(200, json.dumps({"job": jid}))
            if self.path.startswith("/tok"):
                with LOCK:
                    ids = tokens_of(q.get("text", ""))
                return self._s(200, json.dumps({"toks": [tok.decode([i]) for i in ids]}))
            text, pos = q.get("text", ""), int(q.get("pos", -1))
            want = str(q.get("ckpt", ""))
            if want not in set(ADAPTERS.values()):
                # fail loudly rather than silently serving a different checkpoint
                return self._s(400, json.dumps(
                    {"error": "unknown checkpoint %r; expected one of %s"
                              % (want, sorted(set(ADAPTERS.values())))}))
            ad = want
            nn = max(1, min(int(q.get("n", 12)), 48))
            mx = max(4, min(int(q.get("max_new", 16)), 128))
            role = "assistant" if str(q.get("role", "user")) == "assistant" else "user"
            nc = max(1, min(int(q.get("ncar", 3)), GRID.n_car))
            with LOCK:
                vec, ids, pos = state_at(text, pos, role)
                cands = sorted(set(propose(vec, ad, nn, float(q.get("temp", 1.1)), mx)))
                if self.path.startswith("/evolve"):
                    sc = score(cands, vec, nc, mx)
                    cands = list(sc)
                    extra = llm_rewrite(text, ids, pos, sc, nn)
                    cands = sorted(set(cands) | set(extra))
                sc = score(cands, vec, nc, mx)
            rows = [{"phrase": s, "score": v}
                    for s, v in sorted(sc.items(), key=lambda kv: -kv[1])]
            _lbl = next((k for k, v in ADAPTERS.items() if v == ad), ad)
            return self._s(200, json.dumps({"rows": rows, "pos": pos,
                                            "token": tok.decode([ids[pos]]),
                                            "ckpt_used": _lbl,
                                            "secs": round(time.time() - t0, 1)}))
        except Exception as e:
            return self._s(500, json.dumps({"error": "%s: %s" % (type(e).__name__, e)}))


SYS = """You are refining a phrase describing what a language model held in mind at one position.
Scoring: the phrase goes into "Focus on the following idea: <PHRASE> while writing the following
phrase: <a fixed sentence>"; the model writes that sentence and we read its internal state. A phrase
scores well when that state matches the target.
From measurement: naming the mental POSTURE beats restating words ("not searching anymore but found
it" beat paraphrases of "There it is!"); inferring the implicature beats copying ("could go to London
or Paris for a week!" -> "opportunity cost"). Do NOT pad with boilerplate -- an RL run found that
prefixing everything with "not pizza related wording but related ..." scores well and says nothing.
If a phrase would score the same for a completely different text it is worthless.
4-16 words, natural English. Reply with ONE candidate per line, nothing else."""


def llm_rewrite(text, ids, pos, scored, n):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return []
    import anthropic
    cl = anthropic.Anthropic(api_key=key)
    ctx = tok.decode(ids[max(0, pos - 30):pos + 1])
    hist = "\n".join("  %+.4f  %s" % (v, s)
                     for s, v in sorted(scored.items(), key=lambda kv: -kv[1])[:16])
    try:
        r = cl.messages.create(model="claude-sonnet-5", max_tokens=1200,
                               system=[{"type": "text", "text": SYS,
                                        "cache_control": {"type": "ephemeral"}}],
                               messages=[{"role": "user", "content":
                                          "Text ending at the target:\n  ...%s\n\nTarget token %r.\n\n"
                                          "Scored candidates:\n%s\n\nWrite %d new candidates."
                                          % (ctx[-220:], tok.decode([ids[pos]]), hist, n)}])
        body = "".join(getattr(b, "text", "") for b in r.content
                       if getattr(b, "type", "") == "text")
        out = [re.sub(r'^[\s\-\*\d\.\)"]+|"+$', "", x).strip() for x in body.splitlines()]
        return [x for x in out if 3 <= len(x.split()) <= 28][:n]
    except Exception as e:
        print("[s] llm %s: %s" % (type(e).__name__, e), flush=True)
        return []


print("[s] serving on 0.0.0.0:%d" % PORT, flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
