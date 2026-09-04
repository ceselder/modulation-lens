"""Mine deduplicated candidate spans from pretraining text, for a 100M-span harvest.

This is the stage that decides whether the resulting bank is 100M concepts or 40M concepts and 60M
near-duplicates. Vector-space dedup at 100M is 5e15 pairwise comparisons -- impossible -- so the
duplication has to be killed in TEXT space, before a single GPU forward is spent. Three tiers, each
catching what the previous one cannot, all CPU and all cheap:

  1. exact:      normalized-string hash (case, whitespace, punctuation folded)
  2. near-dup:   MinHash over character 5-shingles, banded LSH -> catches "the color of grass is"
                 vs "The colour of the grass is", which tier 1 misses
  3. structural: cap how many spans share a rare-word signature, so one boilerplate template
                 ("All advertised prices exclude tag government fees") cannot contribute 50k spans

Span shape is matched to the existing bank: v3 labels are 2-8 word mid-sentence fragments, mean 4.7
words (46% are 2-4 words, 54% are 5+). We extract contiguous word n-grams from sentence interiors
rather than whole sentences, because that is what the published atoms look like and what the
modulation construction was tuned on.

Deliberately NOT filtered here: whether a span is a "good concept". The whole point of the harvest
is that reliability decides that afterwards, empirically. Filtering on taste up front would remove
exactly the imperative / charged spans that turn out to have the highest steering power.
"""
import os

import modal

app = modal.App("celeste-mine-spans")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("datasets", "pyarrow", "numpy", "huggingface_hub[hf_transfer]", "xxhash",
                    "transformers==5.5.4", "tokenizers")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_ENABLE_HF_TRANSFER": "1",
             "TOKENIZERS_PARALLELISM": "false"}))

WORKER = r'''
import os, re, sys, json, unicodedata
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, xxhash
from datasets import load_dataset
from transformers import AutoTokenizer

SHARD    = int(os.environ["SHARD"])
NSHARD   = int(os.environ["NSHARD"])
TARGET   = int(os.environ["TARGET"])          # spans to emit from this shard
OUT      = os.environ["OUT"]
MINTOK, MAXTOK = 4, 20      # span length in QWEN TOKENS, not words
MINW, MAXW = 2, 14          # word prefilter, generous: token count is the real gate
NUM_PERM, BANDS = 64, 16                      # 16 bands x 4 rows: ~0.75 Jaccard threshold
ROWS = NUM_PERM // BANDS

def h64(x):
    """xxhash requires bytes -- a str raises TypeError, so encode at one place."""
    return xxhash.xxh64_intdigest(x.encode("utf-8", "ignore"))

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

_WS   = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+")
_OK   = re.compile(r"^[\w\s',\-À-ɏͰ-ϿЀ-ӿ]+$", re.UNICODE)

# v3's labels are coherent phrases ("Austrian school of economics", "Valley girl", "Add some
# drama"), not arbitrary word windows. A naive n-gram extractor produces "this imposition of" and
# "copy it to" -- syntactic fragments whose induced direction is dominated by whatever the template
# completes them into. Reliability WOULD eventually filter those, but only after paying to measure
# them, so the boundary check happens here on CPU instead of on 100M GPU forwards.
# Heuristic, not a parser: reject spans that begin or end on a word that cannot close a constituent.
_BAD_END = {
    "a","an","the","this","that","these","those","my","your","his","her","its","our","their",
    "of","in","on","at","to","for","with","by","from","as","into","onto","upon","about","over",
    "under","between","through","during","before","after","and","or","but","nor","so","yet",
    "is","are","was","were","be","been","being","am","has","have","had","do","does","did",
    "will","would","can","could","shall","should","may","might","must","very","more","most",
    "not","no","if","than","then","when","while","because","which","who","whom","whose",
}
_BAD_START = {
    "of","in","on","at","to","for","with","by","from","as","into","onto","upon","and","or",
    "but","nor","so","yet","is","are","was","were","be","been","being","am","has","have","had",
    "does","did","than","then","which","whom","whose","that",
}

def well_formed(span):
    w = span.lower().replace(",", " ").split()
    if len(w) < 2:
        return False
    if w[0] in _BAD_START or w[-1] in _BAD_END:
        return False
    if span[0] in ",-'" or span[-1] in ",-":
        return False
    return True

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    return _WS.sub(" ", s).strip()

# --- MinHash over character 5-shingles ---
rng = np.random.default_rng(0)
A = rng.integers(1, 2**61 - 1, NUM_PERM, dtype=np.int64)
B = rng.integers(0, 2**61 - 1, NUM_PERM, dtype=np.int64)
MP = (1 << 61) - 1

def minhash(s):
    sh = {h64(s[i:i+5]) for i in range(max(len(s) - 4, 1))}
    if not sh: return None
    h = np.fromiter(sh, dtype=np.uint64, count=len(sh)).astype(np.int64) % MP
    return ((np.outer(A, np.ones(len(h), dtype=np.int64)) * h[None, :] + B[:, None]) % MP).min(1)

seen_exact = set()
bands = [dict() for _ in range(BANDS)]
sig_count = {}
out_text, out_meta = [], []
n_seen = n_exact = n_near = n_struct = n_illformed = n_toklen = 0

# Default config (data/, 2410 parquet files) rather than sample-10BT: streaming .shard()
# distributes by FILE, so the shard count cannot exceed the file count without leaving
# containers empty. 2410 files supports the 250-way fan-out; sample-10BT would not.
ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
ds = ds.shard(num_shards=NSHARD, index=SHARD)

for rec in ds:
    if len(out_text) >= TARGET: break
    for sent in _SENT.split(rec.get("text", ""))[:40]:
        if len(out_text) >= TARGET: break
        w = sent.split()
        if len(w) < MINW + 2: continue
        # sentence INTERIORS: skip the first/last word so spans look like the published atoms
        for i in range(1, max(len(w) - MINW, 1)):
            if len(out_text) >= TARGET: break
            # Sweep window sizes in a POSITION-DEPENDENT ORDER. Ascending order plus `break` on
            # first accept systematically picks the shortest window, which gave mean 6.6 tokens and
            # only 0.2% of spans at the 20-token end -- not the 4-20 range asked for. Rotating the
            # order by start position spreads the length distribution without needing RNG state.
            _Ls = (2, 3, 4, 5, 6, 8, 10, 12, 14)
            _off = (i * 7 + len(w)) % len(_Ls)
            for L in _Ls[_off:] + _Ls[:_off]:
                if i + L > len(w) - 1: continue
                span = " ".join(w[i:i+L])
                if not (MINW <= L <= MAXW) or not _OK.match(span): continue
                if not well_formed(span): n_illformed += 1; continue
                if len(span) < 8 or len(span) > 160: continue
                ntok = len(TOK(span, add_special_tokens=False).input_ids)
                if not (MINTOK <= ntok <= MAXTOK): n_toklen += 1; continue
                n_seen += 1
                key = norm(span)
                if not key or len(key.split()) < MINW: continue
                kh = h64(key)
                if kh in seen_exact: n_exact += 1; continue
                # tier 3: rare-word signature, caps boilerplate families
                toks = key.split()
                sig = h64(" ".join(sorted(toks)[:3]))
                if sig_count.get(sig, 0) >= 40: n_struct += 1; continue
                # tier 2: banded LSH
                mh = minhash(key)
                if mh is None: continue
                keys = [xxhash.xxh64_intdigest(mh[b*ROWS:(b+1)*ROWS].tobytes()) for b in range(BANDS)]
                if any(k in bands[b] for b, k in enumerate(keys)): n_near += 1; continue
                seen_exact.add(kh); sig_count[sig] = sig_count.get(sig, 0) + 1
                for b, k in enumerate(keys): bands[b][k] = 1
                out_text.append(span); out_meta.append(ntok)
                break                      # one span per start position

tb = pa.table({"span": pa.array(out_text, pa.string()),
               "n_tokens": pa.array(out_meta, pa.int16()),
               "source": pa.array(["fineweb-edu"] * len(out_text), pa.string())})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pq.write_table(tb, OUT, compression="zstd")
stats = {"shard": SHARD, "emitted": len(out_text), "candidates_seen": n_seen,
         "dropped_exact": n_exact, "dropped_near": n_near, "dropped_structural": n_struct, "dropped_illformed": n_illformed, "dropped_toklen": n_toklen,
         "keep_rate": len(out_text)/max(n_seen,1)}
print("[shard %d] emitted %d of %d candidates (%.1f%%) | dropped exact %d near %d struct %d illformed %d toklen %d"
      % (SHARD, len(out_text), n_seen, 100*stats["keep_rate"], n_exact, n_near, n_struct, n_illformed, n_toklen), flush=True)
print("SPANS_JSON " + json.dumps(stats), flush=True)
sys.stdout.flush(); sys.stderr.flush()
# Skip interpreter teardown. The streaming dataset's background thread races pyarrow at exit and
# aborts with "PyGILState_Release: ... no thread-state for this thread" -- AFTER the parquet is
# safely written, so the data is fine but the process returns -6 (SIGABRT). At 100M spans that
# means thousands of successful shards look like failures to any orchestrator checking exit codes.
os._exit(0)
'''


@app.function(image=img, volumes={"/vol": VOL}, cpu=4.0, memory=32768, timeout=10800,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def mine(shard: int, nshard: int, target: int):
    import subprocess
    out = "/vol/spans/spans-%05d-of-%05d.parquet" % (shard, nshard)
    if os.path.exists(out):
        print("[skip] %s exists" % out); return 0
    open("/root/w.py", "w").write(WORKER)
    p = subprocess.run(["python", "/root/w.py"],
                       env=dict(os.environ, SHARD=str(shard), NSHARD=str(nshard),
                                TARGET=str(target), OUT=out))
    VOL.commit()
    return p.returncode


@app.local_entrypoint()
def main(nshard: int = 250, target: int = 400000):
    """250 shards x 400k = 100M spans. Each shard dedups WITHIN itself; a global exact-hash
    pass afterwards catches cross-shard duplicates (100M int64 hashes ~ 800MB, trivial).
    Cross-shard NEAR-duplicates are not caught -- merging 250 x 16 LSH band tables would
    need ~1.6B entries. Shards read disjoint files, so the residual is web boilerplate
    recurring across documents, which the structural cap partly handles."""
    rcs = list(mine.starmap([(i, nshard, target) for i in range(nshard)]))
    print("return codes:", rcs)
