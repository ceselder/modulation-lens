"""Global exact cross-shard dedup for the FineFineWeb bank.

Each mining shard dedups only WITHIN itself, so identical spans mined by different shards both
survive -- confirmed by "I am so happy" appearing twice with identical rho in the measured
dictionary. The FineWeb-Edu bank got this pass (it removed 5.84% as cross-shard duplicates); the
FineFineWeb bank did not, which is an oversight rather than a design choice.

Keeps the DOMAIN column, which the earlier version dropped.

Each mining shard deduped only WITHIN itself (exact hash + MinHash/LSH + boilerplate cap), because
merging 250 x 16 LSH band tables would need ~1.6B entries. Shards read disjoint FineWeb files, so
what leaks across them is text that recurs in different documents -- boilerplate, quotations,
stock phrasing. This pass removes the exact cross-shard duplicates and reports how many there were,
which is the number that tells us whether near-dup leakage is worth chasing.

Uses a numpy int64 array + np.unique rather than a Python set: 100M hashes is 800MB as an array
versus several GB as a set of boxed ints.

Emits two artifacts:
  spans_dedup/   all unique spans (the durable list, cheap to keep)
  spans_10m/     a uniform random 10M sample, which is what the GPU pass measures first
"""
import os

import modal

app = modal.App("celeste-dedup-ffw")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("pyarrow", "numpy", "xxhash")
       .env({"TOKENIZERS_PARALLELISM": "false"}))

WORKER = r'''
import glob, json, os, re, unicodedata
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, xxhash

N_SAMPLE = int(os.environ.get("N_SAMPLE", "10000000"))
_WS = re.compile(r"\s+")

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    return _WS.sub(" ", s).strip()

IN_DIR = os.environ.get("IN_DIR", "/vol/spans_ffw")
OUT_DIR = os.environ.get("OUT_DIR", "/vol/spans_ffw_dedup")
files = sorted(glob.glob(IN_DIR + "/ffw-*.parquet"))
print("[in] %d shard files" % len(files), flush=True)

spans, ntoks, doms, hashes = [], [], [], []
for i, f in enumerate(files):
    t = pq.read_table(f, columns=["span", "n_tokens", "domain"]).to_pydict()
    spans.extend(t["span"]); ntoks.extend(t["n_tokens"]); doms.extend(t["domain"])
    if (i + 1) % 50 == 0:
        print("   read %d/%d files, %s spans" % (i+1, len(files), "{:,}".format(len(spans))), flush=True)
print("[in] %s spans total" % "{:,}".format(len(spans)), flush=True)

h = np.fromiter((xxhash.xxh64_intdigest(norm(s).encode("utf-8", "ignore")) for s in spans),
                dtype=np.uint64, count=len(spans))
print("[hash] done", flush=True)

_, first_idx = np.unique(h, return_index=True)
first_idx.sort()
n_dup = len(spans) - len(first_idx)
print("[dedup] unique %s | cross-shard exact duplicates removed %s (%.2f%%)"
      % ("{:,}".format(len(first_idx)), "{:,}".format(n_dup), 100*n_dup/len(spans)), flush=True)

sp = np.array(spans, dtype=object)[first_idx]
nt = np.array(ntoks, dtype="int16")[first_idx]
dm = np.array(doms, dtype=object)[first_idx]
del spans, ntoks, doms, h

os.makedirs(OUT_DIR, exist_ok=True)
CH = 2_000_000
for k in range(0, len(sp), CH):
    pq.write_table(pa.table({"span": pa.array(list(sp[k:k+CH]), pa.string()),
                             "n_tokens": pa.array(nt[k:k+CH], pa.int16()),
                             "domain": pa.array(list(dm[k:k+CH]), pa.string()),
                             "source": pa.array(["finefineweb"]*len(sp[k:k+CH]), pa.string())}),
                   OUT_DIR + "/dedup-%05d.parquet" % (k // CH), compression="zstd")
print("[out] wrote %d dedup files" % ((len(sp)+CH-1)//CH), flush=True)

json.dump({"n_in": int(len(sp) + n_dup), "n_unique": int(len(sp)), "n_dup": int(n_dup),
           "dup_pct": 100*n_dup/(len(sp)+n_dup)},
          open("/vol/results_dedup_ffw.json", "w"), indent=1)
print("DEDUP_DONE", flush=True)
'''


@app.function(image=img, volumes={"/vol": VOL}, cpu=8.0, memory=131072, timeout=10800)
def run(n_sample: int = 10_000_000, in_dir: str = "/vol/spans_ffw", out_dir: str = "/vol/spans_ffw_dedup"):
    import subprocess
    open("/root/w.py", "w").write(WORKER)
    rc = subprocess.run(["python", "/root/w.py"],
                        env=dict(os.environ, N_SAMPLE=str(n_sample), IN_DIR=in_dir, OUT_DIR=out_dir)).returncode
    VOL.commit()
    return rc
