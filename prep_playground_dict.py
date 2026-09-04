"""Consolidate the FineFineWeb dictionary into playground-loadable arrays.

Two problems this solves:

1. MEMORY. 10.4M atoms x 5120 fp16 = 106 GB per space, and the playground keeps BOTH raw and
   J-space resident plus a truncated 27B model -- 248 GB against a B200's 180 GB. So the playground
   gets a capped subset, selected by consistency, not the whole bank.
2. LOAD TIME. Reading 519 parquet chunks on every cold start would take many minutes. Consolidated
   .npy memmaps load in seconds.

Selection is by rho_raw descending, subject to a floor, so the subset is the most internally
consistent atoms available -- with the caveat established earlier that the very top of the rho
distribution is spam/boilerplate, so the floor matters more than the ranking. The playground can
then apply tighter filters as masks over the loaded subset at zero cost.
"""
import os

import modal

app = modal.App("celeste-prep-pg-dict")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = modal.Image.debian_slim(python_version="3.12").pip_install("pyarrow", "numpy")

WORKER = r'''
import glob, json, os
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

MD    = os.environ.get("MD", "/vol/rel_5m")
VEC   = os.environ.get("VEC", "/vol/rel_5m_vec")
OUT   = os.environ.get("OUT", "/vol/pg_dict")
CAP   = int(os.environ.get("CAP", "4000000"))
FLOOR = float(os.environ.get("FLOOR", "0.60"))

mdf = sorted(glob.glob(MD + "/*.parquet"))
print("[in] %d metadata chunks" % len(mdf), flush=True)
T = pa.concat_tables([pq.read_table(f) for f in mdf]).to_pydict()
n0 = len(T["span"])
# dedup by span: shards dedup only within themselves
seen, keep = set(), []
for i, s in enumerate(T["span"]):
    if s in seen: continue
    seen.add(s); keep.append(i)
rho = np.array([T["rho"][i] for i in keep], dtype="float32")
print("[in] %s rows -> %s unique | rho mean %.4f" % ("{:,}".format(n0), "{:,}".format(len(keep)), rho.mean()), flush=True)

# ---- initial pruning, before any vectors are read ----
# Three text filters, each targeting a junk class this bank demonstrably contains:
#
#  (a) SPAM. Pharmacy/SEO boilerplate sits at the very top of the rho distribution ("buy
#      sibutramine 10mg" 0.897, "buy soma online with no prescription" 0.886) because it is
#      distinctive and steers consistently. Selecting by rho descending would load it first.
#  (b) FUNCTION-WORD-ONLY spans. The high-agreement set was full of semantically empty
#      connectives -- "commonly referred to", "the emphasis was focused on", "for a sure thing"
#      scored p94-p96 while carrying no concept. Require at least one content word.
#  (c) LENGTH-CORRECTED RANKING. Ranking by raw rho favours nothing by length (rho is flat), but
#      ranking by the AGREEMENT of both spaces did favour short spans. Use the percentile within
#      the span's own length bucket so the subset stays spread over 2-16 tokens.
import re as _re
SPAM = _re.compile(
    r"(sibutramine|diazepam|lorazepam|alprazolam|tramadol|phentermine|zolpidem|ambien|xanax"
    r"|valium|klonopin|adipex|oxycodone|hydrocodone|viagra|cialis|levitra|kamagra|modafinil"
    r"|carisoprodol|clonazepam|meridia|soma\s+(online|pill)|no\s+prescription|without\s+prescription"
    r"|online\s+pharmacy|cheap\s+(pills|meds)|buy\s+\w+\s+(online|mg)|casino|payday\s+loan"
    r"|escort|porn|xxx)", _re.I)
STOP = set("""a an the this that these those my your his her its our their of in on at to for with
by from as into onto upon about over under between through during before after and or but nor so
yet is are was were be been being am has have had do does did will would can could shall should
may might must very more most not no if than then when while because which who whom whose it he
she they we you i there here what how all any both each few many some such only own same too s t
just don now""".split())

def has_content(sv):
    for w in _re.findall(r"[^\W\d_]+", sv.lower()):
        if len(w) >= 3 and w not in STOP:
            return True
    return False

n_spam = n_func = 0
cand = []
for k, i in enumerate(keep):
    if rho[k] < FLOOR: continue
    sv = T["span"][i]
    if SPAM.search(sv): n_spam += 1; continue
    if not has_content(sv): n_func += 1; continue
    cand.append(k)
print("[prune] dropped %s spam-pattern, %s function-word-only -> %s candidates"
      % ("{:,}".format(n_spam), "{:,}".format(n_func), "{:,}".format(len(cand))), flush=True)

# rank by percentile within the span's own token length, so the subset stays length-spread
nt_all = np.array([T["n_tokens"][keep[k]] for k in range(len(keep))], dtype="int16")
pctile = np.zeros(len(keep), dtype="float32")
for Tk in np.unique(nt_all):
    m = np.nonzero(nt_all == Tk)[0]
    o = m[np.argsort(rho[m])]
    pctile[o] = np.arange(len(m)) / max(len(m) - 1, 1) * 100
sel = sorted(cand, key=lambda k: -pctile[k])
sel = sel[:CAP]
sel_idx = [keep[k] for k in sel]
want = {T["span"][i] for i in sel_idx}
print("[sel] floor %.2f + cap %s -> %s atoms (rho %.4f-%.4f)"
      % (FLOOR, "{:,}".format(CAP), "{:,}".format(len(sel_idx)),
         min(rho[k] for k in sel), max(rho[k] for k in sel)), flush=True)
import collections as _c
_lh = _c.Counter(int(T["n_tokens"][i]) for i in sel_idx)
print("[sel] length mix: %s" % dict(sorted(_lh.items())), flush=True)

# vectors, pulled only for the selected spans
vmap = {}
for j, f in enumerate(sorted(glob.glob(VEC + "/*.parquet"))):
    t = pq.read_table(f)
    sps = t.column("span").to_pylist()
    hit = [k for k, s in enumerate(sps) if s in want]
    if hit:
        arr = t.column("vector").combine_chunks()
        V = np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype="float16").reshape(-1, 5120)
        for k in hit: vmap[sps[k]] = V[k]
    if (j + 1) % 100 == 0:
        print("   vec %d files, %s collected" % (j + 1, "{:,}".format(len(vmap))), flush=True)
print("[vec] %s vectors collected" % "{:,}".format(len(vmap)), flush=True)

final = [i for i in sel_idx if T["span"][i] in vmap]
os.makedirs(OUT, exist_ok=True)
V = np.stack([vmap[T["span"][i]] for i in final])
np.save(OUT + "/vectors.npy", V)
for col, dt in (("rho", "float32"), ("rho_j", "float32"), ("S", "float32"),
                ("S_j", "float32"), ("n_tokens", "int16")):
    np.save(OUT + "/%s.npy" % col, np.array([T[col][i] for i in final], dtype=dt))
json.dump([T["span"][i] for i in final], open(OUT + "/labels.json", "w"))
print("[out] %s atoms | vectors %s %.1f GB" %
      ("{:,}".format(len(final)), V.shape, V.nbytes / 1e9), flush=True)
print("PREP_DONE %d" % len(final), flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, cpu=8.0, memory=196608, timeout=10800)
def run(cap: int = 4000000, floor: float = 0.60):
    import subprocess
    open("/root/w.py", "w").write(WORKER)
    rc = subprocess.run(["python", "/root/w.py"],
                        env=dict(os.environ, CAP=str(cap), FLOOR=str(floor))).returncode
    VOL.commit()
    return rc
