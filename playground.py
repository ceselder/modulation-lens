"""Interactive NNOLS playground: decompose an activation into 4 dictionary atoms, with a live
reliability cutoff.

The point is to see what the reliability filter actually costs you. Raising the cutoff shrinks the
dictionary (>=0.80 keeps 12.4% of 1.58M atoms, >=0.85 keeps 2.9%, >=0.90 keeps 0.33%), so the
greedy decomposition has fewer atoms to choose from and the reconstruction gets worse -- but the
atoms it does pick should be more consistent steerers. That tradeoff is the whole open question, and
staring at concrete decompositions is a faster way to form a view than reading survival curves.

Serves a single page: type text, pick a token position, drag the cutoff, get the K atoms greedy
non-negative matching pursuit chose plus their NNLS weights and the fraction of variance explained.
Also sweeps the cutoff automatically so you can see FVE-vs-cutoff for the activation you are
looking at.

Implementation notes:
  * The model is truncated to 43 layers (we read L42) and lm_head is skipped -- verified to give
    bit-identical reads (cosine 1.000000), and it avoids materialising a [B, T, 248320] logits
    tensor we would discard.
  * Atoms live in J-space (A @ J.T) to match how the reward scores compositions.
  * Two-mean centring: atoms are already mean-centred in the release; the target activation has the
    pool mean subtracted, matching the miner.
"""
import os

import modal

app = modal.App("celeste-nnols-playground")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .apt_install("git")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "accelerate", "safetensors",
                    "sentencepiece", "pyarrow", "numpy", "fastapi[standard]",
                    "huggingface_hub[hf_transfer]", "einops", "flash-linear-attention")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       .add_local_dir("/home/celeste/modlens_modal/src", "/root/src", copy=True))

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>NNOLS dictionary playground</title>
<style>
 body{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;max-width:1000px;margin:2rem auto;
      padding:0 1.2rem;background:#faf9f5;color:#191919;line-height:1.5}
 h1{font-size:1.4rem;margin-bottom:.2rem} .sub{color:#87867f;margin-top:0;font-size:.9rem}
 textarea{width:100%;height:5.5rem;font-family:ui-monospace,monospace;font-size:.85rem;padding:.6rem;
          border:1px solid #ddd8cc;border-radius:6px;background:#fff}
 .row{display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap;margin:.8rem 0}
 label{font-size:.85rem;color:#5b5a55} input[type=range]{width:220px}
 button{background:#d97757;color:#fff;border:0;padding:.55rem 1.3rem;border-radius:6px;
        font-size:.9rem;cursor:pointer} button:disabled{opacity:.5;cursor:wait}
 table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.87rem}
 th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #eee7d9}
 th{color:#87867f;font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
 code{background:#f0ece1;padding:.1rem .3rem;border-radius:3px;font-size:.85em}
 .kpi{display:inline-block;margin-right:1.6rem} .kpi b{font-size:1.25rem;color:#d97757}
 .kpi span{display:block;font-size:.72rem;color:#87867f}
 #tokens span{cursor:pointer;padding:.1rem .2rem;border-radius:3px;background:#f0ece1;margin:1px;
              display:inline-block;font-family:ui-monospace,monospace;font-size:.8rem}
 #tokens span.sel{background:#d97757;color:#fff}
 .bar{height:6px;background:#d97757;border-radius:3px;display:inline-block;vertical-align:middle}
 .note{font-size:.8rem;color:#87867f}
</style></head><body>
<h1>NNOLS dictionary playground</h1>
<p class="sub">Decompose a layer-42 activation into dictionary atoms by <b>NNOLS</b> &mdash; greedy matching
pursuit chooses the atoms, exact non-negative least squares (active-set enumeration over all
2<sup>K</sup>&minus;1 supports) refits the weights at every step &mdash; <b>side by side in raw L42
and in J-space</b>. Raise the reliability cutoff to shrink the
dictionary and watch what it costs.</p>

<script>window.addEventListener("load",()=>dictNote());</script>
<textarea id="txt">The knight moved to e4 and he tapped the clock without looking up.</textarea>
<div class="row">
  <button id="tokbtn" onclick="tokenize()">tokenize</button>
  <span class="note">then click a token to read its activation</span>
</div>
<div id="tokens"></div>
<div class="row">
  <label title="thinkies-v3 ships an 8-draw split-half reliability (native median 0.73); it is RESCALED here onto our single-draw pairwise-cosine scale via S=(r/(1-r))/8, giving median 0.254, so one cutoff means the same thing in both banks. The distributions still barely overlap (ours 0.714/0.751/0.832 vs v3 0.192/0.254/0.435) because our bank is pre-filtered at rho&gt;=0.70 and v3 is not -- that is selection, not units. Use per-bank percentile for a fair combined cutoff.">dictionary<br>
    <select id="dict" style="padding:.3rem" onchange="dictNote()">
__DICT_OPTIONS__</select></label>
  <label>consistency cutoff <b id="cutv">0.70</b><span class="note" id="crange"></span><br>
    <input type="range" id="cut" min="0.65" max="0.95" step="0.01" value="0.70"
           oninput="document.getElementById('cutv').textContent=(+this.value).toFixed(2)"></label>
  <label title="Combined mode only. The two banks' rho distributions barely overlap (ours is pre-filtered at 0.70, thinkies-v3 is not), so one absolute cutoff cannot trade them off: below 0.714 it admits all of ours, above 0.435 it deletes all of thinkies. Tick this to read the slider as a within-bank quantile instead, so 0.70 means 'drop the bottom 70% of EACH bank'.">per-bank percentile<br>
    <input type="checkbox" id="pctcut" style="transform:scale(1.3);margin-top:.4rem"
           onchange="sliderBounds()"></label>
  <label>atoms (K)<br><input type="number" id="k" value="4" min="1" max="8" style="width:4rem"></label>
  <label title="Which space the reconstruction is solved in. J is a fitted Jacobian, not a rotation, so the two spaces pick DIFFERENT atoms (measured: 0 of 4 shared) and J-chosen atoms score badly in raw space. Pick one to halve the compute, or both to compare.">reconstruct in<br>
    <select id="sp" style="padding:.3rem"><option value="both">both (compare)</option>
      <option value="raw">raw L42 only</option><option value="jspace">J-space only</option></select></label>
  <button id="go" onclick="run()">decompose</button>
</div>
__PENDING__
<p class="note" id="dnote"></p>
<div id="out"></div>

<script>
const DR = __DICT_RANGES__;
let POS = null;
function sliderBounds(){
  const _dk=document.getElementById('dict').value;
  const usePct=(_dk==='combined' && document.getElementById('pctcut').checked);
  const c=document.getElementById('cut');
  if(usePct){
    c.min=0; c.max=0.95; c.step=0.01; c.value=0.70;
    document.getElementById('cutv').textContent='0.70';
    document.getElementById('crange').textContent=
      ' (percentile \u2014 keeps the top 30% of EACH bank)';
    return;
  }
  // slider bounds from the selected dictionary's real percentiles (p2 / median / p98)
  const r=DR[_dk];
  if(r){
    c.min=r[0]; c.max=r[2]; c.step=Math.max(0.005,((r[2]-r[0])/60).toFixed(3));
    c.value=r[1]; document.getElementById('cutv').textContent=(+r[1]).toFixed(3);
    document.getElementById('crange').textContent=
      ' (this bank spans '+r[0].toFixed(2)+'\u2013'+r[2].toFixed(2)+', median '+r[1].toFixed(2)+')';}
}
function dictNote(){
  const d=document.getElementById('dict').value;
  // Combined mode DEFAULTS to per-bank percentile. Measured: an absolute cutoff of 0.70 leaves
  // thinkies-v3 with 17 survivors of 1.58M, so "combined" silently degenerates to ours-only.
  document.getElementById('pctcut').checked = (d==='combined');
  sliderBounds();
  document.getElementById('dnote').innerHTML = d==='finefineweb'
    ? 'FineFineWeb: 4M atoms selected from an 11.6M bank &mdash; uniform 2&ndash;16 tokens, 67 domains, '
      +'digits and HTML boilerplate removed. Statistic is the single-draw pairwise cosine (median 0.71 raw, 0.78 J).'
    : d==='thinkies-v3'
    ? 'thinkies-v3: 1.58M atoms, pre-filtered at 0.65. Native statistic is the 8-draw split-half '
      +'reliability (median 0.73), rescaled here onto the pairwise-cosine scale so the cutoff is comparable.'
    : 'BOTH banks searched together (5.6M atoms). thinkies-v3 reliability is rescaled onto the '
      +'pairwise-cosine scale via S = (r/(1-r))/8, so the SCALE matches &mdash; but the '
      +'distributions barely overlap, because our bank is pre-filtered at rho&gt;=0.70 and v3 is '
      +'not. An absolute 0.70 therefore leaves v3 with <b>17 atoms of 1.58M</b>. Per-bank '
      +'percentile is on by default here for that reason; it is what makes the two banks '
      +'comparable. The per-atom <b>src</b> column shows which bank each pick came from.';
}
async function tokenize(){
  const b=document.getElementById('tokbtn'); b.disabled=true; b.textContent='working...';
  const r=await fetch('tokenize',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text:document.getElementById('txt').value})});
  b.disabled=false; b.textContent='tokenize';
  if(!r.ok){document.getElementById('out').innerHTML='<p style="color:#b04">tokenize failed (HTTP '
     +r.status+') -- container starting, try again.</p>'; return;}
  const d=await r.json();
  const c=document.getElementById('tokens'); c.innerHTML='';
  d.tokens.forEach((t,i)=>{const s=document.createElement('span');
    s.textContent=t.replace(/ /g,'\\u00b7'); s.onclick=()=>{POS=i;
      [...c.children].forEach(x=>x.className=''); s.className='sel';}; c.appendChild(s);});
  POS=d.tokens.length-1; c.children[POS].className='sel';
}
async function run(){
  if(POS===null){await tokenize();}
  const g=document.getElementById('go'); g.disabled=true; g.textContent='working...';
  document.getElementById('out').innerHTML='<p class="note">Running. The FIRST request after idle '
    +'cold-starts a GPU container and loads the 27B model plus 1.58M atoms in both spaces, which '
    +'takes about 3 minutes. Later requests are seconds.</p>';
  let r;
  try{
  r=await fetch('decompose',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:document.getElementById('txt').value, pos:POS,
      cutoff:+document.getElementById('cut').value,
      k:+document.getElementById('k').value,
      space:document.getElementById('sp').value,
      pct_cut:document.getElementById('pctcut').checked,
      dict:document.getElementById('dict').value})});
  }catch(e){ g.disabled=false; g.textContent='decompose';
    document.getElementById('out').innerHTML='<p style="color:#b04">request failed: '+e
      +'<br>If this was the first request, the container may still be starting -- try again.</p>';
    return; }
  let d;
  try{ d=await r.json(); }
  catch(e){ g.disabled=false; g.textContent='decompose';
    document.getElementById('out').innerHTML='<p style="color:#b04">bad response (HTTP '+r.status
      +') -- container probably still warming up. Try again in a minute.</p>'; return; }
  g.disabled=false; g.textContent='decompose';
  if(d.error){document.getElementById('out').innerHTML='<p style="color:#b04">'+d.error+'</p>';return;}
  const S=d.spaces, X=d.cross, keys=Object.keys(S);
  let _pb='', _warn='';
  if(d.per_bank_survivors){
    const _ks=Object.keys(d.per_bank_survivors);
    _pb=' ('+_ks.map(function(n){
      return n+': '+d.per_bank_survivors[n].toLocaleString()
        +(d.thresholds?' @rho'+d.thresholds[n]:'');}).join(', ')+')';
    const _starved=_ks.filter(function(n){return d.per_bank_survivors[n]<1000;});
    if(_starved.length) _warn='<p class="note" style="color:#b04"><b>'+_starved.join(', ')
      +'</b> has almost no atoms left at this cutoff, so this is effectively a single-bank run. '
      +'Tick <b>per-bank percentile</b> to filter each bank on its own distribution.</p>';
  } else if(d.n_atoms<1000){
    _warn='<p class="note" style="color:#b04">Only <b>'+d.n_atoms.toLocaleString()+'</b> atoms '
      +'survive this cutoff'+(d.threshold!==undefined?' (rho&gt;='+d.threshold+')':'')+'. A '
      +'handful of atoms still returns a plausible-looking FVE, so treat this as unreliable '
      +'rather than as a result &mdash; lower the cutoff or tick per-bank percentile.</p>';
  }
  const DHDR='<p class="note"><b>'+d.dict+'</b> &mdash; '+d.dict_size.toLocaleString()+' atoms loaded, '
    +d.n_atoms.toLocaleString()+' at or above the cutoff'+_pb
    +(d.pct_cut?' &mdash; cutoff read as a WITHIN-BANK percentile':'')+'.</p>'+_warn;
  const NAME={raw:'raw L42', jspace:'J-space'};
  let h=DHDR+'<p class="note">'
   +(X?('Greedy run independently in each space; <b>'+X.overlap+' of '+X.k
        +'</b> chosen atoms are shared.'):'')+'</p>';
  h+='<table><tr><th></th>'+keys.map(k=>'<th>'+NAME[k]+'</th>').join('')+'</tr>'
   +'<tr><td>FVE</td>'+keys.map(k=>'<td><b>'+S[k].fve.toFixed(3)+'</b></td>').join('')+'</tr>'
   +'<tr><td>cosine</td>'+keys.map(k=>'<td>'+S[k].cos.toFixed(3)+'</td>').join('')+'</tr>'
   +'<tr><td>best single atom</td>'+keys.map(k=>'<td>'+S[k].best_single.toFixed(3)+'</td>').join('')+'</tr>';
  if(X){h+='<tr><td>its atoms scored in the OTHER space</td><td>'+X.raw_atoms_in_jspace.toFixed(3)
        +'</td><td>'+X.jspace_atoms_in_raw.toFixed(3)+'</td></tr>';}
  h+='</table><div class="row" style="align-items:flex-start">';
  keys.forEach(k=>{h+='<div style="flex:1;min-width:320px"><b style="font-size:.9rem">'+NAME[k]
      +'</b><table><tr><th>#</th><th>atom</th><th>w</th><th>rel</th>'
      +(d.src_mix?'<th>src</th>':'')+'</tr>';
    S[k].atoms.forEach((a,i)=>{h+='<tr><td>'+(i+1)+'</td><td>'+a.label+'</td><td>'
      +a.w.toFixed(3)+'</td><td>'+a.rel.toFixed(3)+'</td>'
      +(a.src?('<td><code>'+(a.src==='finefineweb'?'ours':'v3')+'</code></td>'):'')+'</tr>';});
    h+='</table></div>';});
  h+='</div><table><tr><th>cutoff</th><th>atoms</th>'
   +keys.map(k=>'<th>FVE '+NAME[k]+'</th>').join('')
   +keys.map(k=>'<th>all '+ (document.getElementById('k').value) +' atoms ('+NAME[k]+')</th>').join('')+'</tr>';
  d.sweep.forEach(r=>{h+='<tr><td>'+r.cutoff.toFixed(2)+'</td><td>'+r.n.toLocaleString()+'</td>'
    +keys.map(k=>'<td>'+((k=='raw'?r.fve_raw:r.fve_j)||0).toFixed(3)+'</td>').join('')
    +keys.map(k=>'<td>'+((k=='raw'?r.top_raw:r.top_j)||'-')+'</td>').join('')+'</tr>';});
  const fmt=a=>a.map(x=>'<code>'+x.tok.replace(/ /g,'\u00b7')+'</code> '
      +(100*x.p).toFixed(1)+'%').join(' &middot; ');
  h+='</table><table><tr><th>reference readout at this token</th><th>top vocabulary tokens</th></tr>'
   +'<tr><td>J-lens &nbsp;<span class="note">lm_head(norm(J&middot;h<sub>42</sub>))</span></td><td>'
   +fmt(d.jlens_tokens)+'</td></tr>'
   +'<tr><td>logit lens at L42 &nbsp;<span class="note">no J</span></td><td>'
   +fmt(d.logitlens_tokens)+'</td></tr></table>'
   +'<p class="note">The J-lens is the baseline this project is trying to beat: it scores 0.256 on '
   +'workspace-bench where trained lenses reach 0.44-0.52. Useful as a sanity check that the '
   +'activation you picked carries what you think it does.</p>';
  h+='<p class="note">Reliability is always the RAW-space value shipped with the dictionary: '
   +'it is agreement across 16 template draws, and the release keeps only only the mean vector for each atom, '
   +'not the individual draws, so a J-space reliability cannot be recomputed from it. That would need '
   +'re-measuring all 1.58M atoms (~2.4 GPU-days).</p>';
  document.getElementById('out').innerHTML=h;
}
</script></body></html>"""

WORKER_INIT = None


@app.cls(image=img, volumes={"/vol": VOL}, gpu="B200:1", timeout=3600,
         scaledown_window=3600, min_containers=1)
class Playground:
    @modal.enter()
    def load(self):
        import glob
        import sys
        import numpy as np
        import pyarrow.parquet as pq
        import torch
        import torch.nn as nn
        sys.path.insert(0, "/root/src")
        import inv_core as C
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.makedirs("/workspace", exist_ok=True)
        if not os.path.exists("/workspace/.hf_home"):
            os.symlink("/vol/.hf_home", "/workspace/.hf_home")
        self.C = C
        self.torch = torch
        self.np = np
        self.dev = "cuda"
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
        m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to("cuda").eval()
        self.inner = m.model
        # hold explicit references: `m` is local, and the truncated base model would not keep
        # lm_head alive. J-lens readout = lm_head(final_norm(J @ h42)).
        self.lm_head = m.lm_head
        self.final_norm = getattr(m.model, "norm", None) or getattr(m.model, "final_layernorm")
        self.inner.layers = nn.ModuleList(list(self.inner.layers[:43]))
        self.J = C.load_jlens(42, "cuda")
        self.hook = {"h": None}
        self.inner.layers[42].register_forward_hook(
            lambda mm, i, o: self.hook.__setitem__("h", o[0] if isinstance(o, tuple) else o))

        # ---- dictionary 2: the FineFineWeb bank, consolidated to .npy so a cold start is seconds
        # rather than parsing 500+ parquet chunks. Capped at 4M because 10.4M x 5120 fp16 is 106 GB
        # per space and this holds BOTH spaces plus the model.
        self.dicts = {}
        if os.path.exists("/vol/pg_dict/vectors.npy"):
            import json as _j
            Vf = np.load("/vol/pg_dict/vectors.npy", mmap_mode="r")
            lf = _j.load(open("/vol/pg_dict/labels.json"))
            rf = np.load("/vol/pg_dict/rho.npy")
            rfj = np.load("/vol/pg_dict/rho_j.npy")
            # DROP atoms over MAXTOK tokens. Mining sampled 2-16 uniformly, but the 13-16 tail
            # reads as two clauses awkwardly concatenated rather than one unit, so those atoms are
            # not usable as a vocabulary however consistently they steer (user call 2026-09-04).
            # Applied at load, since pg_dict ships n_tokens alongside the vectors.
            MAXTOK = 12
            _nt = np.load("/vol/pg_dict/n_tokens.npy")
            _sel = np.nonzero(_nt <= MAXTOK)[0]
            if len(_sel) < len(lf):
                print("[pg] length filter <=%d tok: %d -> %d atoms (%.1f%% kept)"
                      % (MAXTOK, len(lf), len(_sel), 100.0 * len(_sel) / max(len(lf), 1)),
                      flush=True)
                # Do NOT do Vf = Vf[_sel]: fancy-indexing the mmap would materialise ~30 GB of
                # fp16 in CPU RAM. Keep Vf lazy and gather the surviving rows per slice in the
                # projection loop below.
                lf = [lf[i] for i in _sel]
                rf, rfj = rf[_sel], rfj[_sel]
            # CHUNKED. Materialising 4M x 5120 in fp32 is 82 GB, and a full-size A @ J.T another
            # 82 GB -- with the model (~36 GB) and thinkies (~32 GB) that is ~240 GB against a
            # 180 GB B200. Preallocate the two fp16 outputs (41 GB each) and fill them in slices,
            # so peak extra memory is one slice rather than the whole bank.
            N2 = len(_sel)
            A2n = torch.empty((N2, 5120), dtype=torch.float16, device="cuda")
            A2J = torch.empty((N2, 5120), dtype=torch.float16, device="cuda")
            CH2 = 200_000
            for a in range(0, N2, CH2):
                blk = torch.from_numpy(
                    np.ascontiguousarray(Vf[_sel[a:a+CH2]])).to("cuda", torch.float32)
                A2n[a:a+CH2] = (blk / blk.norm(dim=1, keepdim=True).clamp(min=1e-8)).half()
                bj = blk @ self.J.T
                A2J[a:a+CH2] = (bj / bj.norm(dim=1, keepdim=True).clamp(min=1e-8)).half()
                del blk, bj
            torch.cuda.empty_cache()
            print("[pg] finefineweb projected in %d-row slices (peak slice %.1f GB)"
                  % (CH2, CH2 * 5120 * 4 / 1e9), flush=True)
            self.dicts["finefineweb"] = {
                "A": A2n, "AJ": A2J, "labels": lf,
                "rel": torch.from_numpy(rf).to("cuda"),
                "rel_j": torch.from_numpy(rfj).to("cuda"),
            }
            print("[pg] finefineweb dict: %d atoms, rho %.3f-%.3f"
                  % (len(lf), float(rf.min()), float(rf.max())), flush=True)
            torch.cuda.empty_cache()

        # fp16 matvecs against the banks must accumulate in fp32 over 5120 dims, or the
        # shortlist itself becomes unreliable.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        labs, vecs, rels = [], [], []
        for sh in sorted(glob.glob("/vol/thinkies_v3/thinkies_v3-*-of-00007.parquet")):
            for b in pq.ParquetFile(sh).iter_batches(batch_size=16384,
                                                     columns=["label", "vector", "reliability"]):
                labs += b.column("label").to_pylist()
                rels.append(np.asarray(b.column("reliability").to_numpy(zero_copy_only=False), dtype="float32"))
                vecs.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                                       dtype="float32").reshape(-1, 5120))
        Vt = np.concatenate(vecs)
        del vecs
        # Same <=12-token cap as our bank. thinkies-v3 ships no n_tokens, so tokenise the labels
        # (1.58M short strings, seconds). Applying it to only ONE bank would hand the unfiltered
        # bank an edge on long targets in a combined search, which would corrupt exactly the
        # head-to-head this tool exists to make.
        _enc = self.tok(labs, add_special_tokens=False)["input_ids"]
        _tsel = np.array([i for i, e in enumerate(_enc) if len(e) <= 12], dtype=np.int64)
        del _enc
        if len(_tsel) < len(labs):
            print("[pg] thinkies length filter <=12 tok: %d -> %d atoms (%.1f%% kept)"
                  % (len(labs), len(_tsel), 100.0 * len(_tsel) / max(len(labs), 1)), flush=True)
            labs = [labs[i] for i in _tsel]
            rels = [np.concatenate(rels)[_tsel]]
            # Vt stays whole (it is already 32 GB of fp32 in CPU RAM); the projection loop below
            # gathers only the surviving rows, one slice at a time.
        else:
            _tsel = np.arange(len(labs), dtype=np.int64)
        # BOTH spaces stay resident so raw-vs-J is a side-by-side, not two separate runs.
        # J is a fitted Jacobian, not a rotation, so cos(Jx,Jy) != cos(x,y) and the two
        # reconstructions can genuinely disagree about which atoms are best.
        #
        # CHUNKED for the same reason as the finefineweb bank above, and this path is why the
        # container OOMed at 172.9/178.3 GB: the unchunked version put a 32 GB fp32 copy of the
        # bank on the GPU, then a 32 GB temporary for the normalise, then another 32 GB for
        # A @ J.T -- ~96 GB of transients on top of the ~120 GB already held by the model and
        # the 4M bank. Filling preallocated fp16 buffers in slices keeps the resident cost at
        # the 2 x 16 GB we actually need and the transient cost at one slice.
        Nt = len(_tsel)
        self.A = torch.empty((Nt, 5120), dtype=torch.float16, device="cuda")
        self.AJ = torch.empty((Nt, 5120), dtype=torch.float16, device="cuda")
        CHT = 200_000
        for a in range(0, Nt, CHT):
            blk = torch.from_numpy(
                np.ascontiguousarray(Vt[_tsel[a:a+CHT]])).to("cuda", torch.float32)
            self.A[a:a+CHT] = (blk / blk.norm(dim=1, keepdim=True).clamp(min=1e-8)).half()
            bj = blk @ self.J.T
            self.AJ[a:a+CHT] = (bj / bj.norm(dim=1, keepdim=True).clamp(min=1e-8)).half()
            del blk, bj
        del Vt
        torch.cuda.empty_cache()
        print("[pg] thinkies projected in %d-row slices, %d atoms" % (CHT, Nt), flush=True)
        self.labels = labs
        self.rel = torch.from_numpy(np.concatenate(rels)).to("cuda")
        # Every shipped atom has this raw-L42 mean subtracted. The target activation must have it
        # subtracted too: raw L42 is ~93% a shared constant, so an UNCENTRED target makes every
        # cosine dominated by that constant, and whichever atom best aligns with it wins regardless
        # of content. This project measured the size of the effect directly -- one shared mean puts
        # a blank string at 0.259, two means put it at 0.008.
        self.ref_mean = torch.from_numpy(
            np.load("/vol/thinkies_v3/thinkies_v3_ref_mean.npy").astype("float32")).to("cuda")
        print("[pg] ref_mean loaded, norm %.2f" % float(self.ref_mean.norm()), flush=True)
        # thinkies-v3 ships an 8-DRAW SPLIT-HALF reliability; our bank stores the SINGLE-DRAW
        # pairwise cosine. Both are transforms of the same signal-to-noise S, so convert v3 onto our
        # scale -- otherwise a single cutoff silently means two different things in a combined bank:
        #     rel_8 = 8S/(8S+1)  =>  S = (rel/(1-rel))/8  =>  rho = S/(S+1)
        _r8 = self.rel.clamp(1e-6, 1 - 1e-6)
        _S = (_r8 / (1 - _r8)) / 8.0
        self.rel_thinkies_rho = (_S / (1 + _S)).float()
        print("[pg] thinkies reliability rescaled to the pairwise-cosine scale: median %.4f (was %.4f)"
              % (float(self.rel_thinkies_rho.median()), float(self.rel.median())), flush=True)
        self.dicts["thinkies-v3"] = {"A": self.A, "AJ": self.AJ, "labels": self.labels,
                                     "rel": self.rel_thinkies_rho, "rel_j": None,
                                     "rel_native": self.rel}
        print("[pg] %d atoms in J-space, reliability %.3f-%.3f"
              % (len(labs), float(self.rel.min()), float(self.rel.max())), flush=True)
        torch.cuda.empty_cache()

    def _act(self, text, pos):
        """-> (raw L42 activation, same activation pushed through J)."""
        torch = self.torch
        ids = self.tok(text, add_special_tokens=False).input_ids
        pos = max(0, min(pos, len(ids) - 1))
        with torch.no_grad():
            self.inner(input_ids=torch.tensor([ids], device="cuda"))
            h = self.hook["h"].float()[0, pos]
        hc = h.float() - self.ref_mean          # centre to match the atoms
        return hc, (hc @ self.J.T).float()

    def _vocab(self, vec, n=8):
        """Top-n vocabulary tokens for a residual-stream vector, via final norm + LM head."""
        torch = self.torch
        with torch.no_grad():
            x = self.final_norm(vec.to(self.lm_head.weight.dtype).unsqueeze(0))
            lg = self.lm_head(x).float().squeeze(0)
            p = lg.softmax(-1)
            v, i = p.topk(n)
        return [{"tok": self.tok.decode([int(t)]), "p": float(x)} for x, t in zip(v.tolist(), i.tolist())]

    def _nnls(self, B, t):
        """Exact non-negative least squares by active-set enumeration, matching the reward's
        nnls_small. For k<=8 there are at most 255 supports, so enumerating them and keeping the
        lowest-residual FEASIBLE one is exact. Clamping an unconstrained lstsq is NOT the NNLS
        solution -- zeroing the negative coefficients without refitting the survivors gives a
        strictly worse fit -- and negative weights are meaningless for a readout anyway, since
        "minus <phrase>" is not something a lens can report.
        """
        torch = self.torch
        n = B.shape[0]
        best_w, best_res = None, float("inf")
        for mask in range(1, 1 << n):
            idx = [k for k in range(n) if (mask >> k) & 1]
            S = B[idx].T
            sol = torch.linalg.lstsq(S, t.unsqueeze(1)).solution.squeeze(1)
            if bool((sol < -1e-8).any()):
                continue
            res = float((t - S @ sol).norm())
            if res < best_res:
                w = torch.zeros(n, device=B.device, dtype=sol.dtype)
                w[idx] = sol
                best_w, best_res = w, res
        return best_w if best_w is not None else torch.zeros(n, device=B.device, dtype=B.dtype)

    def _greedy_multi(self, target, banks, k):
        """Greedy MP + exact NNLS over SEVERAL atom banks at once.

        The banks stay separate in GPU memory and only the per-atom similarity vectors are
        concatenated (a few million floats), so combining 1.58M + 4M atoms costs no extra
        memory -- a real concatenation of the [N, 5120] fp16 tensors would cost ~57 GB per space.
        Returns picks as (bank_index, row) so the UI can show which dictionary each atom came from.
        """
        torch = self.torch
        t = target / target.norm().clamp(min=1e-8)
        chosen = []          # list of (bank, row)
        resid = t.clone()
        for _ in range(k):
            best = None
            for bi, (atoms, m) in enumerate(banks):
                # NO GATHER. atoms[idx] built a fresh [survivors, 5120] fp16 tensor and .float()
                # doubled it -- 38 GB for a 2M-survivor mask, which OOMed the container at
                # decompose time. A matvec needs no gather: score every atom in fp16 (output is
                # N floats) and put the mask on the OUTPUT instead.
                if not bool(m.any()): continue
                sims = (atoms @ resid.half()).float()
                sims.masked_fill_(~m, float("-inf"))
                for pb, pr in chosen:
                    if pb == bi: sims[pr] = float("-inf")
                # The fp16 matvec resolves cosines to ~5e-4, enough to shortlist but able to
                # mis-order near-ties, so re-score the top 64 exactly in fp32 -- 64 rows is free.
                top = torch.topk(sims, min(64, sims.numel())).indices
                top = top[sims[top] > float("-inf")]
                if not top.numel(): continue
                ex = atoms[top].float() @ resid
                jj = int(ex.argmax()); sj = float(ex[jj])
                if best is None or sj > best[0]:
                    best = (sj, bi, int(top[jj]))
            if best is None or best[0] <= 0: break
            chosen.append((best[1], best[2]))
            B = torch.stack([banks[b][0][r].float() for b, r in chosen])
            w = self._nnls(B, t)
            resid = t - w @ B
        if not chosen:
            return [], torch.zeros(0, device="cuda"), 0.0
        B = torch.stack([banks[b][0][r].float() for b, r in chosen])
        w = self._nnls(B, t)
        rec = w @ B
        cos = float((rec @ t) / rec.norm().clamp(min=1e-8))
        return chosen, w, cos

    def _greedy(self, target, mask, k, AJ=None):
        """Greedy non-negative matching pursuit, then exact NNLS on the chosen support.

        Plain greedy MP: no mutual-similarity ceiling on the picks. Matching pursuit suppresses
        near-duplicates implicitly anyway, because each pick is scored against the RESIDUAL, and an
        atom close to one already chosen has little residual left to explain.
        """
        torch = self.torch
        t = target / target.norm().clamp(min=1e-8)
        AJ = self.AJ if AJ is None else AJ
        resid = t.clone()
        chosen = []
        for _ in range(k):
            # NO GATHER -- see _greedy_multi. AJ[idx_pool].float() materialised the whole
            # surviving subset in fp32 (38 GB at a 2M mask) purely to compute N dot products.
            sims = (AJ @ resid.half()).float()
            sims.masked_fill_(~mask, float("-inf"))
            for pr in chosen:
                sims[pr] = float("-inf")
            top = torch.topk(sims, min(64, sims.numel())).indices
            top = top[sims[top] > float("-inf")]
            if not top.numel():
                break
            ex = AJ[top].float() @ resid
            jj = int(ex.argmax())
            if float(ex[jj]) <= 0:
                break
            gi = int(top[jj])
            chosen.append(gi)
            B = AJ[chosen].float()
            w = self._nnls(B, t)
            resid = t - w @ B
        if not chosen:
            return [], torch.zeros(0, device="cuda"), 0.0
        B = AJ[chosen].float()
        w = self._nnls(B, t)
        rec = w @ B
        cos = float((rec @ t) / rec.norm().clamp(min=1e-8))
        return chosen, w, cos

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse
        api = FastAPI()

        @api.get("/")
        def root():
            names = [n for n in ("finefineweb", "thinkies-v3") if n in self.dicts]
            import json as _js
            rng = {}
            for n in names:
                r = self.dicts[n]["rel"].float()
                q = self.torch.quantile(
                    r[:: max(1, r.numel() // 200000)],
                    self.torch.tensor([0.02, 0.5, 0.98], device=r.device))
                rng[n] = [round(float(x), 3) for x in q]
            if len(names) > 1:
                lo = min(rng[n][0] for n in names); hi = max(rng[n][2] for n in names)
                rng["combined"] = [round(lo, 3), round(
                    float(sum(rng[n][1] for n in names) / len(names)), 3), round(hi, 3)]
            opts = []
            if len(names) > 1:
                tot = sum(len(self.dicts[n]["labels"]) for n in names)
                opts.append('<option value="combined">BOTH combined (%.1fM)</option>' % (tot / 1e6))
            for n in names:
                sz = len(self.dicts[n]["labels"])
                lbl = "FineFineWeb (%.1fM, ours)" % (sz / 1e6) if n == "finefineweb" \
                    else "thinkies-v3 (%.2fM)" % (sz / 1e6)
                opts.append('<option value="%s">%s</option>' % (n, lbl))
            pend = "" if "finefineweb" in self.dicts else (
                '<p class="note" style="color:#b06">FineFineWeb bank still consolidating &mdash; '
                'only thinkies-v3 is loaded right now. Reload once it lands to get the 4M bank and '
                'the combined option.</p>')
            return HTMLResponse(PAGE.replace("__DICT_OPTIONS__", "\n      ".join(opts))
                                    .replace("__PENDING__", pend)
                                    .replace("__DICT_RANGES__", _js.dumps(rng)))

        @api.post("/tokenize")
        async def _tok(r: Request):
            d = await r.json()
            ids = self.tok(d["text"], add_special_tokens=False).input_ids
            return JSONResponse({"tokens": [self.tok.decode([i]) for i in ids]})

        @api.post("/decompose")
        async def _dec(r: Request):
            d = await r.json()
            try:
                t_raw, t_j = self._act(d["text"], int(d["pos"]))
                k = int(d.get("k", 4))
                cut = float(d["cutoff"])
                which = d.get("dict", "thinkies-v3")
                if which == "combined" and len(self.dicts) > 1:
                    order = [n for n in ("thinkies-v3", "finefineweb") if n in self.dicts]
                    torch = self.torch
                    # The two banks live on DISJOINT rho scales -- ours is pre-filtered at 0.70 in
                    # both spaces (p2/median/p98 = 0.714/0.751/0.832) while thinkies-v3 ships its
                    # full unfiltered distribution (0.192/0.254/0.435). A single absolute cutoff
                    # therefore cannot trade the banks off: below 0.714 it admits all of ours, above
                    # 0.435 it deletes all of thinkies. pct_cut reads the slider as a WITHIN-BANK
                    # quantile instead, so "top 30%" means top 30% of each bank.
                    pct_cut = bool(d.get("pct_cut", False))
                    masks, thr_used = {}, {}
                    for n in order:
                        rel = self.dicts[n]["rel"]
                        if pct_cut:
                            sub = rel[:: max(1, rel.numel() // 200000)].float()
                            thr = float(torch.quantile(sub, min(max(cut, 0.0), 1.0)))
                        else:
                            thr = cut
                        masks[n] = rel >= thr
                        thr_used[n] = round(thr, 4)
                    def _run(space_key, tgt):
                        banks = [(self.dicts[n][space_key], masks[n]) for n in order]
                        return self._greedy_multi(tgt, banks, k)
                    def _best1(space_key, tgt):
                        """Max single-atom cosine over BOTH banks -- the baseline the k-atom
                        composition has to beat. Was hardcoded to 0.0 on this path, which silently
                        removed the only reference point for judging the composition."""
                        tt = tgt / tgt.norm().clamp(min=1e-8)
                        b = 0.0
                        for n in order:
                            m = masks[n]
                            if not bool(m.any()): continue
                            At = self.dicts[n][space_key]
                            sc = (At @ tt.half()).float()
                            sc.masked_fill_(~m, float("-inf"))
                            top = torch.topk(sc, min(64, sc.numel())).indices
                            top = top[sc[top] > float("-inf")]
                            if not top.numel(): continue
                            b = max(b, float((At[top].float() @ tt).max()))
                        return b
                    res = {"dict": "combined", "space_mode": d.get("space", "both"),
                           "dict_size": sum(len(self.dicts[n]["labels"]) for n in order),
                           "pct_cut": pct_cut, "thresholds": thr_used,
                           "n_atoms": int(sum(int(masks[n].sum()) for n in order)),
                           "per_bank_survivors": {n: int(masks[n].sum()) for n in order},
                           "spaces": {}, "sweep": [],
                           "jlens_tokens": self._vocab((t_raw + self.ref_mean) @ self.J.T),
                           "logitlens_tokens": self._vocab(t_raw + self.ref_mean)}
                    want2 = d.get("space", "both")
                    todo2 = [("raw", "A", t_raw), ("jspace", "AJ", t_j)]
                    if want2 in ("raw", "jspace"):
                        todo2 = [x for x in todo2 if x[0] == want2]
                    for nm, sk, tgt in todo2:
                        ch, w, cos = _run(sk, tgt)
                        res["spaces"][nm] = {
                            "fve": cos*cos, "cos": cos, "best_single": _best1(sk, tgt),
                            "atoms": [{"label": self.dicts[order[b]]["labels"][r],
                                       "w": float(x), "rel": float(self.dicts[order[b]]["rel"][r]),
                                       "src": order[b]}
                                      for (b, r), x in zip(ch, w.tolist())]}
                    # per-bank share of the picks: the whole point of combining
                    allp = [a["src"] for sp in res["spaces"].values() for a in sp["atoms"]]
                    res["src_mix"] = {n: allp.count(n) for n in order}
                    return JSONResponse(res)
                D = self.dicts.get(which) or self.dicts["thinkies-v3"]
                LAB = D["labels"]; REL = D["rel"]; RELJ = D["rel_j"]
                A_, AJ_ = D["A"], D["AJ"]
                # thinkies-v3 ships an 8-draw split-half reliability; our bank stores the
                # single-draw pairwise cosine. Different scales for the same underlying quantity,
                # so a cutoff means different things -- label it in the UI rather than convert.
                # pct_cut applies here too. It used to be honoured ONLY on the combined branch
                # and ignored SILENTLY everywhere else, which made a scripted single-bank
                # comparison run at an absolute cutoff without saying so -- thinkies-v3 at an
                # absolute 0.70 has 17 atoms of 1.58M, and a 17-atom run still returns a
                # perfectly plausible-looking FVE.
                if bool(d.get("pct_cut", False)):
                    _sub = REL[:: max(1, REL.numel() // 200000)].float()
                    _thr = float(self.torch.quantile(_sub, min(max(cut, 0.0), 1.0)))
                else:
                    _thr = cut
                mask = REL >= _thr

                def fve_in(space_atoms, target, chosen):
                    """FVE of a FIXED atom set, refit by NNLS in the given space. Lets us score the
                    J-chosen atoms in raw space and vice versa -- the comparison that says whether
                    the choice of space actually matters or only relabels the numbers."""
                    if not chosen: return 0.0
                    tt = target / target.norm().clamp(min=1e-8)
                    B = space_atoms[chosen].float()
                    wv = self._nnls(B, tt)
                    rec = wv @ B
                    c = float((rec @ tt) / rec.norm().clamp(min=1e-8))
                    return c * c

                want = d.get("space", "both")
                todo = [("raw", A_, t_raw), ("jspace", AJ_, t_j)]
                if want in ("raw", "jspace"):
                    todo = [x for x in todo if x[0] == want]
                res = {"n_atoms": int(mask.sum()), "spaces": {}, "space_mode": want, "dict": which,
                       "dict_size": len(LAB),
                       "pct_cut": bool(d.get("pct_cut", False)), "threshold": round(_thr, 4),
                       # reference readouts for the same position: the J-lens (J*h42 pushed through
                       # the head, i.e. what the Jacobian lens predicts the model will say) and the
                       # plain logit lens at L42 (h42 straight through the head, no J).
                       # vocabulary readouts use the UNCENTRED residual: lm_head expects a real
                       # residual-stream vector, not a mean-subtracted direction.
                       "jlens_tokens": self._vocab((t_raw + self.ref_mean) @ self.J.T),
                       "logitlens_tokens": self._vocab(t_raw + self.ref_mean)}
                picks = {}
                for name, atoms, tgt in todo:
                    ch, w, cos = self._greedy(tgt, mask, k, AJ=atoms)
                    picks[name] = ch
                    tt = tgt / tgt.norm()
                    best = float((atoms[ch].float() @ tt).max()) if ch else 0.0
                    res["spaces"][name] = {
                        "fve": cos*cos, "cos": cos, "best_single": best,
                        "atoms": [{"label": LAB[c], "w": float(x), "rel": float(REL[c])}
                                  for c, x in zip(ch, w.tolist())]}
                # cross-evaluation only makes sense when both were run
                if len(picks) == 2:
                    res["cross"] = {
                        "raw_atoms_in_jspace": fve_in(AJ_, t_j, picks["raw"]),
                        "jspace_atoms_in_raw": fve_in(A_, t_raw, picks["jspace"]),
                        "overlap": len(set(picks["raw"]) & set(picks["jspace"])),
                        "k": k}
                sweep = []
                for c2 in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
                    m2 = REL >= c2
                    n2 = int(m2.sum())
                    if n2 < k:
                        sweep.append({"cutoff": c2, "n": n2, "fve_raw": 0.0, "fve_j": 0.0,
                                      "top_raw": "(too few atoms)", "top_j": "(too few atoms)"})
                        continue
                    row = {"cutoff": c2, "n": n2}
                    if any(x[0] == "raw" for x in todo):
                        cr, _, kr = self._greedy(t_raw, m2, k, AJ=A_)
                        row["fve_raw"] = kr*kr
                        row["top_raw"] = " / ".join(LAB[c] for c in cr) if cr else "-"
                    if any(x[0] == "jspace" for x in todo):
                        cj, _, kj = self._greedy(t_j, m2, k, AJ=AJ_)
                        row["fve_j"] = kj*kj
                        row["top_j"] = " / ".join(LAB[c] for c in cj) if cj else "-"
                    sweep.append(row)
                res["sweep"] = sweep
                return JSONResponse(res)
            except Exception as e:
                return JSONResponse({"error": "%s: %s" % (type(e).__name__, str(e)[:200])})

        return api
