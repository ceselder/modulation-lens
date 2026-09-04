"""Assemble the pruned dictionary: apply a consistency floor and emit a clean artifact.

The consistency statistic rho is used as a FLOOR, not a top-slice selector. Measured on 85k of the
1M atoms, the top of the distribution is numeric quantity phrases and pharmacy boilerplate ("buy
sibutramine 10mg" at 0.897, "at least 8" at 0.893) -- rho rewards steering strength, and spam
boilerplate steers very consistently. The BOTTOM is what rho identifies usefully: context-dependent
fragments ("sealing work is", "effective because the wheels") that cannot serve as atoms. So the
filter's job is to remove the floor, not to select the ceiling.

Emits, per tier, a metadata table plus its vectors, and prints the domain/length composition of each
so any skew introduced by filtering is visible rather than assumed.
"""
import os

import modal

app = modal.App("celeste-assemble-pruned")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("pyarrow", "numpy"))

WORKER = r'''
import glob, json, os, collections
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

# No defaults. These are passed as FUNCTION ARGUMENTS by run() below, not inherited from a
# local shell -- a local `MD=... modal run` does NOT cross into the container, and when these
# had defaults that failure mode silently assembled the wrong bank and overwrote its report.
MD  = os.environ["MD"]
VEC = os.environ["VEC"]
OUT = os.environ["OUT"]
FLOORS = [float(x) for x in os.environ["FLOORS"].split(",")]
EXPECT = int(os.environ.get("EXPECT_CHUNKS", "0"))
# Spans longer than this are dropped. Mining sampled 2-16 tokens uniformly, but the 13-16 token
# end is not coherent enough to serve as an atom (user call, 2026-09-04) -- they read as two
# clauses awkwardly concatenated rather than one unit. 12 also matches the bullet length used
# elsewhere in the project.
MAXTOK = int(os.environ.get("MAX_TOKENS", "12"))
# BAN newline / CR / tab. 5.14% of mined spans contain one, and they are layout boilerplate that
# spans a line break rather than a phrase: 'Sale price\nPickup', 'TEXTS\nPrice',
# 'on Current Events\nOn August'. Same objection as multi-sentence concatenation -- two fragments
# glued together is not one unit, however consistently it steers. It also broke bullet round-trips
# (joining atoms with \n and splitting on \n tore 11.6% of them in two).
BAN_WS = os.environ.get("BAN_WS", "1") == "1"
print("[cfg] MD=%s VEC=%s OUT=%s FLOORS=%s" % (MD, VEC, OUT, FLOORS), flush=True)

mdf = sorted(glob.glob(MD + "/*.parquet"))
vcf = sorted(glob.glob(VEC + "/*.parquet"))
print("[in] %d metadata chunks, %d vector chunks" % (len(mdf), len(vcf)), flush=True)
if not mdf:
    raise SystemExit("no metadata chunks at %s -- refusing to assemble an empty bank" % MD)
if len(mdf) != len(vcf):
    raise SystemExit("stats/vector desync: %d vs %d chunks" % (len(mdf), len(vcf)))
if EXPECT and len(mdf) < EXPECT:
    raise SystemExit("partial input: %d of %d expected chunks -- measurement is still running"
                     % (len(mdf), EXPECT))
tabs = [pq.read_table(f) for f in mdf]
T = pa.concat_tables(tabs).to_pydict()
# Dedup by span FIRST. Each mining shard deduped only within itself, so identical spans mined by
# different shards both got measured -- 4.61% of the bank, confirmed by a global pass. Keeping the
# first occurrence: the duplicates are the same text with the same vector, so which one survives
# does not matter, only that one does.
_seen, _keep = set(), []
for i, sv in enumerate(T["span"]):
    if sv in _seen: continue
    _seen.add(sv); _keep.append(i)
n_before = len(T["span"])
T = {k: [v[i] for i in _keep] for k, v in T.items()}
n = len(T["span"])
print("[dedup] %s -> %s unique atoms (%.2f%% duplicate)"
      % ("{:,}".format(n_before), "{:,}".format(n), 100*(n_before-n)/n_before), flush=True)
rho = np.array(T["rho"], dtype="float32")
print("[in] %s atoms | rho mean %.4f sd %.4f" % ("{:,}".format(n), rho.mean(), rho.std()), flush=True)

# ---- decide WHAT TO KEEP before loading any vectors ----------------------------------------
# vmap is a dict of 11.57M separate numpy arrays: 118 GB of data plus ~112 bytes of object
# overhead each, and with the metadata lists on top the container was killed at 128 GB and got
# preempted three times at 320 GB. The filters below need only metadata, so apply them first and
# insert vectors for the SURVIVORS only: 8.03M instead of 11.57M, ~85 GB, back inside 128 GB.
_nt_pre = np.array(T["n_tokens"], dtype="int32")
_ws_pre = np.array([("\n" in x) or ("\r" in x) or ("\t" in x) for x in T["span"]]) if BAN_WS \
    else np.zeros(len(T["span"]), dtype=bool)
_keep_pre = (_nt_pre <= MAXTOK) & (~_ws_pre)
KEEPSET = {T["span"][i] for i in np.nonzero(_keep_pre)[0]}
print("[pre] %s of %s atoms survive the <=%d-token and whitespace filters (%.1f%%) -- only their "
      "vectors are loaded" % ("{:,}".format(len(KEEPSET)), "{:,}".format(len(_nt_pre)), MAXTOK,
                              100.0 * len(KEEPSET) / max(len(_nt_pre), 1)), flush=True)

# vectors, keyed by span so a chunk-order mismatch cannot silently misalign them
vmap = {}
for f in sorted(glob.glob(VEC + "/*.parquet")):
    t = pq.read_table(f)
    arr = t.column("vector").combine_chunks()
    V = np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype="float16").reshape(-1, 5120)
    for s, v in zip(t.column("span").to_pylist(), V):
        if s in KEEPSET:
            vmap[s] = v
print("[in] %s vectors loaded" % "{:,}".format(len(vmap)), flush=True)

# domain comes from the span bank, not the reliability tables -- join it back
dom = {}
for f in sorted(glob.glob("/vol/spans_ffw/*.parquet")):
    t = pq.read_table(f, columns=["span", "domain"]).to_pydict()
    for s, d in zip(t["span"], t["domain"]):
        dom.setdefault(s, d)
print("[in] domain labels for %s spans" % "{:,}".format(len(dom)), flush=True)

# Store EVERY filter variant as a column. Measuring costs ~1.3 GPU-days; filtering costs seconds,
# so committing to one rule now would be the expensive mistake. Variants:
#   rho, rho_j                     raw statistics in each space
#   rho_per_token(_j)              rho / n_tokens -- note a GLOBAL cut on this keeps ONLY the
#                                  shortest length, because rho is flat across length so length
#                                  becomes the whole signal (measured: top 20% was 100% 4-token)
#   rho_pct_in_len(_j)             percentile WITHIN the span's own length bucket -- length-neutral,
#                                  so "top 20%" means top 20% at every length
#   agree_pct                      min(rho_pct_in_len, rho_j_pct_in_len): high only if BOTH spaces
#                                  rank it well, which is the agreement filter, length-corrected
rho_a = np.array(T["rho"], dtype="float32")
rj_a  = np.array(T["rho_j"], dtype="float32")
nt_a  = np.array(T["n_tokens"], dtype="float32")
rpt   = rho_a / np.maximum(nt_a, 1)
rpt_j = rj_a  / np.maximum(nt_a, 1)
pct   = np.zeros(len(rho_a), dtype="float32")
pct_j = np.zeros(len(rho_a), dtype="float32")
for T_ in np.unique(nt_a):
    m = np.nonzero(nt_a == T_)[0]
    for src, dst in ((rho_a, pct), (rj_a, pct_j)):
        o = m[np.argsort(src[m])]
        dst[o] = np.arange(len(m)) / max(len(m) - 1, 1) * 100
agree = np.minimum(pct, pct_j)
print("[cols] added rho_per_token, rho_pct_in_len, agree_pct (length-neutral agreement)", flush=True)

os.makedirs(OUT, exist_ok=True)
report = {}
if BAN_WS:
    _ws = _ws_pre
    _nws = int(_ws.sum())
    print("[ws] dropping %s of %s atoms containing newline/CR/tab (%.2f%%)"
          % ("{:,}".format(_nws), "{:,}".format(len(_ws)), 100.0 * _nws / max(len(_ws), 1)),
          flush=True)
else:
    _ws = _ws_pre

_long = int((nt_a > MAXTOK).sum())
print("[len] dropping %s of %s atoms over %d tokens (%.1f%%)"
      % ("{:,}".format(_long), "{:,}".format(len(nt_a)), MAXTOK, 100.0 * _long / max(len(nt_a), 1)),
      flush=True)
for floor in FLOORS:
    keep = np.nonzero((rho >= floor) & (nt_a <= MAXTOK) & (~_ws))[0]
    keep = np.array([i for i in keep if T["span"][i] in vmap], dtype=np.int64)
    tag = "all" if floor <= 0 else ("f%03d" % int(floor * 100))
    spans = [T["span"][i] for i in keep]
    cols = {
        "span": pa.array(spans, pa.string()),
        "n_tokens": pa.array([T["n_tokens"][i] for i in keep], pa.int16()),
        "domain": pa.array([dom.get(s, "unknown") for s in spans], pa.string()),
        "rho": pa.array([T["rho"][i] for i in keep], pa.float32()),
        "ci_lo": pa.array([T["ci_lo"][i] for i in keep], pa.float32()),
        "ci_hi": pa.array([T["ci_hi"][i] for i in keep], pa.float32()),
        "S": pa.array([T["S"][i] for i in keep], pa.float32()),
        "rho_j": pa.array([T["rho_j"][i] for i in keep], pa.float32()),
        "S_j": pa.array([T["S_j"][i] for i in keep], pa.float32()),
        "n_cells": pa.array([T["n_cells"][i] for i in keep], pa.int16()),
        "rho_per_token": pa.array(rpt[keep], pa.float32()),
        "rho_per_token_j": pa.array(rpt_j[keep], pa.float32()),
        "rho_pct_in_len": pa.array(pct[keep], pa.float32()),
        "rho_j_pct_in_len": pa.array(pct_j[keep], pa.float32()),
        "agree_pct": pa.array(agree[keep], pa.float32()),
    }
    pq.write_table(pa.table(cols), "%s/meta_%s.parquet" % (OUT, tag), compression="zstd")
    V = np.stack([vmap[s] for s in spans]) if spans else np.zeros((0, 5120), "float16")
    CH = 250000
    for k in range(0, max(len(V), 1), CH):
        if not len(V): break
        pq.write_table(pa.table({
            "span": pa.array(spans[k:k+CH], pa.string()),
            "vector": pa.array(list(V[k:k+CH]), pa.list_(pa.float16(), 5120)),
        }), "%s/vec_%s_%02d.parquet" % (OUT, tag, k // CH), compression="zstd")
    lens = collections.Counter(int(T["n_tokens"][i]) for i in keep)
    doms = collections.Counter(dom.get(s, "unknown") for s in spans)
    r = np.array([T["rho"][i] for i in keep], dtype="float32")
    report[tag] = {"floor": floor, "n": len(keep), "pct": 100*len(keep)/n,
                   "rho_mean": float(r.mean()) if len(r) else 0.0,
                   "n_domains": len(doms), "len_hist": dict(sorted(lens.items())),
                   "top_domains": doms.most_common(5)}
    print("\n[%s] floor %.2f -> %s atoms (%.1f%% of %s) | rho mean %.4f | %d domains"
          % (tag, floor, "{:,}".format(len(keep)), 100*len(keep)/n, "{:,}".format(n),
             report[tag]["rho_mean"], len(doms)), flush=True)
    print("    length mix: %s" % report[tag]["len_hist"], flush=True)
    print("    top domains: %s" % report[tag]["top_domains"], flush=True)

json.dump(report, open(OUT + "/report.json", "w"), indent=1)
print("[out] report at %s/report.json" % OUT, flush=True)
print("\nASSEMBLE_DONE", flush=True)
'''


# memory=131072 was NOT enough and Modal killed the runner mid-write: the script loads EVERY
# vector, and 11,571,229 x 5120 x fp16 is 118.5 GB against a 128 GB request, before the per-tier
# copies. 320 GB clears it with room. (The cheaper fix is to stream vectors per tier instead of
# loading the whole bank, but that is a rewrite and this runs once.)
@app.function(image=img, volumes={"/vol": VOL}, cpu=8.0, memory=180224, timeout=10800)
def run(md: str, vec: str, out: str,
        floors: str = "0.0,0.65,0.70,0.75", expect_chunks: int = 0,
        max_tokens: int = 12):
    import subprocess
    os.makedirs(out, exist_ok=True)
    open("/root/w.py", "w").write(WORKER)
    rc = subprocess.run(["python", "/root/w.py"],
                        env=dict(os.environ, MD=md, VEC=vec, OUT=out, FLOORS=floors,
                                 EXPECT_CHUNKS=str(expect_chunks),
                                 MAX_TOKENS=str(max_tokens))).returncode
    VOL.commit()
    return rc
