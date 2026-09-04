"""Mine span candidates from FineFineWeb (m-a-p/FineFineWeb).

Chosen over FineWeb-edu because its 66,103 JSONL files are organised BY DOMAIN (aerospace, news,
health, economics, law, sports, ~40 more) and every record carries a `domain` field. A concept
dictionary wants domain breadth, and sharding by stride across the file list gives each container a
domain mix for free, so the bank does not end up dominated by journalism.

Span rule, deliberately simple:
  * split into sentences, take contiguous word windows ANYWHERE in the sentence (edges included)
  * keep 4-16 Qwen tokens
  * no punctuation inside the span (apostrophes allowed -- "person's" is one word)
  * NO function-word boundary filter: a span may start or end on "of"/"the"/"is"
  * rotate window length by position so the token-length distribution spreads instead of
    collapsing onto the shortest accepted window

Dedup is the part that stays aggressive, because it is what stops the bank being 60% near-copies:
  1. exact:      normalized-string hash (case/punct/whitespace folded)
  2. near-dup:   MinHash over character 5-shingles, 16 bands x 4 rows (~0.75 Jaccard)
  3. structural: cap spans sharing a rare-word signature, so one boilerplate template cannot
                 contribute thousands

Why filtering matters even though mining is cheap: mining runs at ~128,000 spans/s across CPU
containers, while MEASURING a span's consistency runs at 7.5 spans/s/GPU. Measurement is ~2000x
slower, so anything removed on CPU is removed 2000x more cheaply than it would be paid for on GPU.
"""
import os

import modal

app = modal.App("celeste-mine-ffw")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("pyarrow", "numpy", "huggingface_hub[hf_transfer]", "xxhash",
                    "transformers==5.5.4", "tokenizers")
       .env({"HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false"}))

WORKER = r'''
import json, os, re, sys, unicodedata
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, xxhash
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

SHARD  = int(os.environ["SHARD"]); NSHARD = int(os.environ["NSHARD"])
TARGET = int(os.environ["TARGET"]); OUT = os.environ["OUT"]
MINTOK, MAXTOK = 2, 16      # 1-token atoms come from the COMPLETE vocab set, not mining
NUM_PERM, BANDS = 64, 16
ROWS = NUM_PERM // BANDS

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
_WS   = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+")
# letters / digits / spaces / apostrophes only -- no commas, dashes, brackets, quotes
# No digits: numeric quantity phrases ("at least 8", "a minimum of 6", "with a maximum of 14")
# came out at the very TOP of the consistency distribution, so they would dominate any
# high-rho slice while carrying no concept. Letters/space/apostrophe only.
_OK   = re.compile(r"^[^\W\d_][\w\s'À-ɏͰ-ϿЀ-ӿ]*$", re.UNICODE)
_DIGIT = re.compile(r"\d")
# HTML boilerplate glued onto the preceding word by tag-stripping in the source: "veinsRead More",
# "wettypeRead More In 1982", "to PreviousDwelling". A NARROW blocklist rather than a general
# case-join ban -- case joins are only 0.88% of spans and mostly legitimate camelCase brand names
# (BitMEX, CanvasDecoArt, ProAmp, DesiTude), so banning them would delete real product names to
# remove ~0.3% of genuine garbage. These phrases have near-zero false-positive rate.
_TAGJUNK = re.compile(
    r"(read\s*more|continue\s+reading|click\s+here|view\s+more|learn\s+more|see\s+more"
    r"|show\s+more|share\s+this|posted\s+(on|by)|filed\s+under|tagged\s+with|comments?\s+off"
    r"|previous(post|dwelling)|next\s*post|skip\s+to\s+content|leave\s+a\s+reply"
    r"|subscribe\s+to|sign\s+up\s+for|all\s+rights\s+reserved|privacy\s+policy"
    r"|terms\s+of\s+(use|service)|add\s+to\s+cart|out\s+of\s+stock)", re.I)

def h64(x): return xxhash.xxh64_intdigest(x.encode("utf-8", "ignore"))

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    return _WS.sub(" ", s).strip()

rng = np.random.default_rng(0)
A = rng.integers(1, 2**61 - 1, NUM_PERM, dtype=np.int64)
B = rng.integers(0, 2**61 - 1, NUM_PERM, dtype=np.int64)
MP = (1 << 61) - 1

def minhash(s):
    sh = {h64(s[i:i+5]) for i in range(max(len(s) - 4, 1))}
    h = np.fromiter(sh, dtype=np.uint64, count=len(sh)).astype(np.int64) % MP
    return ((A[:, None] * h[None, :] + B[:, None]) % MP).min(1)

fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
# The file list is CACHED on the volume. Listing it is a recursive tree walk over 66,103 files, and
# 100 containers doing that at once gets every single one HTTP 429'd. Built once by
# build_file_list() before the fan-out.
files = json.load(open("/vol/ffw_files.json"))
# INTERLEAVE BY DOMAIN before sharding. The file list is sorted alphabetically, so a plain
# files[SHARD::NSHARD] stride makes shard i START at file i -- and `aerospace` alone has enough
# files that shards 0..~200 all begin inside it. Since a single 317MB file supplies the whole
# per-shard target, the first run mined 10M spans that were 100% aerospace.
by_dom = {}
for f in files:
    by_dom.setdefault(f.split("/")[3], []).append(f)
order, k = [], 0
while any(k < len(v) for v in by_dom.values()):
    for d in sorted(by_dom):
        if k < len(by_dom[d]): order.append(by_dom[d][k])
    k += 1
mine = order[SHARD::NSHARD]
# and cap spans per FILE so a shard cannot satisfy its whole quota from one domain
PER_FILE = int(os.environ.get('PER_FILE', '2000'))   # absolute, so shard size does not
                                                     # concentrate spans on few files
print("[shard %d] %d files, %d domains in my slice, <=%d spans/file"
      % (SHARD, len(mine), len({f.split("/")[3] for f in mine}), PER_FILE), flush=True)

seen, bands, sig_count = set(), [dict() for _ in range(BANDS)], {}
QUOTA = {T: TARGET // (MAXTOK - MINTOK + 1) for T in range(MINTOK, MAXTOK + 1)}
filled = {T: 0 for T in QUOTA}
out_s, out_t, out_d = [], [], []
n_raw = n_ok = n_tok = n_ex = n_near = n_str = n_tag = 0
_Ls = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)

for path in mine:
    if all(filled[t] >= QUOTA[t] for t in QUOTA): break
    dom = path.split("/")[3]
    file_start = len(out_s)
    fh = None
    for attempt in range(5):
        try:
            fh = fs.open(path, "r", encoding="utf-8"); break
        except Exception as e:
            if attempt == 4:
                print("   open failed %s: %s" % (path, type(e).__name__), flush=True)
            else:
                import time as _t, random as _r
                _t.sleep(min(60, 2 ** attempt) * (1 + _r.random()))
    if fh is None:
        continue
    with fh as f:
        for line in f:
            if len(out_s) >= TARGET: break
            if len(out_s) - file_start >= PER_FILE: break   # move on to the next domain
            try: rec = json.loads(line)
            except Exception: continue
            if rec.get("lang") != "en": continue
            for sent in _SENT.split(rec.get("text", ""))[:60]:
                if len(out_s) >= TARGET: break
                # Work in TOKEN space: tokenize the sentence ONCE, then take contiguous token
                # windows. Length is then exactly T by construction, so the bank can be made
                # UNIFORM over 2-16 tokens instead of skewed short (word-window sampling gave
                # mean 7.4 with mode 4, because short words yield few tokens). Also faster: one
                # tokenization per sentence rather than one per candidate span.
                ids = TOK(sent, add_special_tokens=False).input_ids
                if len(ids) < MINTOK + 1: continue
                for i in range(0, len(ids)):
                    if len(out_s) >= TARGET: break
                    if len(out_s) - file_start >= PER_FILE: break
                    # cycle the target length so coverage of 2..16 is even across start positions
                    T = MINTOK + ((i * 7 + len(ids)) % (MAXTOK - MINTOK + 1))
                    if filled[T] >= QUOTA[T]:
                        # this length is done; try the least-filled length at this position instead
                        T = min(filled, key=lambda t: filled[t])
                        if filled[T] >= QUOTA[T]: continue
                    if i + T > len(ids): continue
                    span = TOK.decode(ids[i:i+T]).strip()
                    n_raw += 1
                    if not span or len(span) > 130: continue
                    if _DIGIT.search(span) or not _OK.match(span): n_ok += 1; continue
                    if _TAGJUNK.search(span): n_tag += 1; continue
                    # decode->encode must round-trip to the same length, or n_tokens is a lie
                    if len(TOK(span, add_special_tokens=False).input_ids) != T:
                        n_tok += 1; continue
                    key = norm(span)
                    if not key: continue
                    kh = h64(key)
                    if kh in seen: n_ex += 1; continue
                    toks = key.split()
                    sg = h64(" ".join(sorted(toks)[:3]))
                    if sig_count.get(sg, 0) >= 40: n_str += 1; continue
                    mh = minhash(key)
                    bk = [h64(mh[b*ROWS:(b+1)*ROWS].tobytes().hex()) for b in range(BANDS)]
                    if any(k in bands[b] for b, k in enumerate(bk)): n_near += 1; continue
                    seen.add(kh); sig_count[sg] = sig_count.get(sg, 0) + 1
                    for b, k in enumerate(bk): bands[b][k] = 1
                    out_s.append(span); out_t.append(T); out_d.append(dom)
                    filled[T] += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pq.write_table(pa.table({"span": pa.array(out_s, pa.string()),
                         "n_tokens": pa.array(out_t, pa.int16()),
                         "domain": pa.array(out_d, pa.string()),
                         "source": pa.array(["finefineweb"]*len(out_s), pa.string())}),
               OUT, compression="zstd")
import collections as _c
_lc = _c.Counter(out_t)
st = {"shard": SHARD, "emitted": len(out_s), "quota_per_len": TARGET // (MAXTOK - MINTOK + 1), "raw": n_raw, "drop_punct": n_ok,
      "drop_roundtrip": n_tok, "drop_tagjunk": n_tag, "drop_exact": n_ex, "drop_near": n_near, "drop_struct": n_str,
      "keep_of_raw": len(out_s)/max(n_raw,1), "domains": len(set(out_d)),
      "len_hist": {int(k): int(v) for k, v in sorted(_lc.items())}}
print("[shard %d] emitted %d of %d raw (%.1f%%) | punct %d tagjunk %d roundtrip %d exact %d near %d struct %d | %d domains | len 2-16 counts %s"
      % (SHARD, len(out_s), n_raw, 100*st["keep_of_raw"], n_ok, n_tag, n_tok, n_ex, n_near, n_str,
         st["domains"], [st["len_hist"].get(t, 0) for t in range(2, 17)]), flush=True)
print("FFW_JSON " + json.dumps(st), flush=True)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)   # skip interpreter teardown: the fs reader thread races pyarrow and aborts with -6
'''


@app.function(image=img, volumes={"/vol": VOL}, cpu=2.0, timeout=3600,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def build_file_list():
    """List FineFineWeb's 66k files ONCE and cache to the volume, so the fan-out does not 429."""
    import json as _j
    from huggingface_hub import HfFileSystem as _FS
    if os.path.exists("/vol/ffw_files.json"):
        n = len(_j.load(open("/vol/ffw_files.json")))
        print("[list] cached: %d files" % n)
        return n
    fs = _FS(token=os.environ["HF_TOKEN"])
    files = sorted(fs.glob("datasets/m-a-p/FineFineWeb/*/*.jsonl"))
    _j.dump(files, open("/vol/ffw_files.json", "w"))
    VOL.commit()
    doms = sorted({f.split("/")[3] for f in files})
    print("[list] %d files across %d domains" % (len(files), len(doms)))
    return len(files)


@app.function(image=img, volumes={"/vol": VOL}, cpu=4.0, memory=32768, timeout=10800,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def mine(shard: int, nshard: int, target: int, outdir: str = "/vol/spans_ffw"):
    import subprocess
    out = outdir + "/ffw-%05d-of-%05d.parquet" % (shard, nshard)
    if os.path.exists(out):
        print("[skip] %s" % out); return 0
    open("/root/w.py", "w").write(WORKER)
    rc = subprocess.run(["python", "/root/w.py"],
                        env=dict(os.environ, SHARD=str(shard), NSHARD=str(nshard),
                                 TARGET=str(target), OUT=out)).returncode
    VOL.commit()
    return rc


@app.local_entrypoint()
def main(nshard: int = 100, target: int = 100000, outdir: str = "/vol/spans_ffw"):
    """Smoke: 100 shards x 100k = 10M spans."""
    print("file list ready: %s files" % build_file_list.remote())
    rcs = list(mine.starmap([(i, nshard, target, outdir) for i in range(nshard)]))
    ok = sum(1 for r in rcs if r == 0)
    print("shards ok: %d/%d" % (ok, len(rcs)))
