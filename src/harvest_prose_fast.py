#!/usr/bin/env python3
"""Batched L42 harvest. Same positions and labels as harvest_prose.py, ~5x the throughput.

harvest_prose.py runs ONE document per forward pass, which is latency-bound on a 27B: 500k
positions at 6 per document is 83k single-sequence forwards, about 2.8h. Two observations make
batching exact rather than approximate:

  * CPOST (the assistant-turn opener that followed the passage) is causally INVISIBLE to every
    position inside the passage, so dropping it changes no harvested activation. That removes the
    only reason the passage had to sit at a fixed offset from the end.
  * With CPOST gone, documents can be RIGHT-padded: position len(CPRE)+k holds passage position k
    for every row regardless of that row's length, and nothing after k can influence it -- true for
    causal attention and equally true for the recurrent GDN layers, whose state at k depends only on
    tokens up to k. (Left-padding would corrupt the recurrent state and is not used.)

Everything else -- chat-native read, boundary-token positions, clause-start labels -- is copied
verbatim so the output is interchangeable with the existing pool.
"""
import argparse, glob, os, random, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="", help="parquet with a 'text' column; blank = HF fineweb stream")
ap.add_argument("--layer", type=int, default=42)
ap.add_argument("--n", type=int, default=500000)
ap.add_argument("--per-doc", type=int, default=6)
ap.add_argument("--max-tok", type=int, default=256)
ap.add_argument("--min-pos", type=int, default=24)
ap.add_argument("--label-tok", type=int, default=16)
ap.add_argument("--at-boundary", type=int, default=1)
ap.add_argument("--batch", type=int, default=24, help="documents per forward pass")
ap.add_argument("--out", default="/workspace/inv/data/prose_L42_500k.parquet")
ap.add_argument("--skip-docs", type=int, default=0,
                help="discard this many leading documents before harvesting. Streaming fineweb "
                     "yields the SAME document order in every process, so parallel shards without "
                     "distinct offsets would harvest the same documents at different positions -- "
                     "highly correlated activations dressed up as a bigger pool.")
ap.add_argument("--seed", type=int, default=1,
                help="1, not 0: seed 0 produced the existing 60k pool and we want fresh documents")
A = ap.parse_args()
random.seed(A.seed)
BASE = "Qwen/Qwen3.6-27B"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
G = {}
model.model.layers[A.layer].register_forward_hook(
    lambda m, i, o: G.__setitem__("h", o[0] if isinstance(o, tuple) else o))

_r = tok.apply_chat_template([{"role": "user", "content": "XSLOT"}], tokenize=False,
                             add_generation_prompt=True, enable_thinking=False)
_a, _b = _r.split("XSLOT")
CPRE = tok(_a, add_special_tokens=False).input_ids
NPRE = len(CPRE)
PAD = tok.pad_token_id if tok.pad_token_id is not None else 248046
print("[hp] chat wrapper: %d prefix tokens (suffix dropped: causally invisible) | pad %d"
      % (NPRE, PAD), flush=True)

DELIM = set()
for _s in ". , ! ? ; : ... —".split():
    for _c in (_s, " " + _s):
        _i = tok(_c, add_special_tokens=False).input_ids
        if len(_i) == 1:
            DELIM.add(_i[0])
print("[hp] %d delimiter token ids" % len(DELIM), flush=True)


skipped = [0]


def docs():
    if A.corpus:
        for f in sorted(glob.glob(A.corpus)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["text"]):
                for t in b.to_pydict()["text"]:
                    if skipped[0] < A.skip_docs:
                        skipped[0] += 1
                        continue
                    yield t
    else:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train",
                          streaming=True, token=os.environ.get("HF_TOKEN"))
        if A.skip_docs:
            ds = ds.skip(A.skip_docs)
        for r in ds:
            yield r["text"]


D = 5120
schema = pa.schema([("activation_vector", pa.list_(pa.float32(), D)),
                    ("label", pa.string()), ("ctx", pa.string()),
                    ("doc_id", pa.int64()), ("pos", pa.int64())])
w = pq.ParquetWriter(A.out, schema)
buf_v, buf_l, buf_c, buf_d, buf_p = [], [], [], [], []
got, di, t0 = 0, 0, time.time()
FLUSH = 8192


def flush():
    global buf_v, buf_l, buf_c, buf_d, buf_p
    w.write_table(pa.table({
        "activation_vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.stack(buf_v).reshape(-1), type=pa.float32()), D),
        "label": pa.array(buf_l), "ctx": pa.array(buf_c),
        "doc_id": pa.array(buf_d, type=pa.int64()),
        "pos": pa.array(buf_p, type=pa.int64())}, schema=schema))
    buf_v, buf_l, buf_c, buf_d, buf_p = [], [], [], [], []


def run(pending):
    """pending: list of (doc_id, ids). Right-pad, one forward, extract each doc's positions."""
    global got
    L = max(len(x[1]) for x in pending)
    B = len(pending)
    inp = torch.full((B, NPRE + L), PAD, dtype=torch.long)
    msk = torch.zeros(B, NPRE + L, dtype=torch.long)
    for r, (_, ids) in enumerate(pending):
        inp[r, :NPRE] = torch.tensor(CPRE)
        inp[r, NPRE:NPRE + len(ids)] = torch.tensor(ids)
        msk[r, :NPRE + len(ids)] = 1
    model(input_ids=inp.to("cuda"), attention_mask=msk.to("cuda"))
    H = G["h"].float()
    for r, (doc_id, ids) in enumerate(pending):
        cand = list(range(A.min_pos, len(ids)))
        if A.at_boundary:
            cand = [k for k in cand if ids[k] in DELIM]
        random.shuffle(cand)
        for k in cand[: A.per_doc]:
            buf_v.append(H[r, NPRE + k].cpu().numpy())
            lo = max(0, k - A.label_tok + 1)
            for j in range(k - 1, lo - 1, -1):
                if ids[j] in DELIM:
                    lo = j + 1
                    break
            buf_l.append(tok.decode(ids[lo:k + 1]).strip())
            buf_c.append(tok.decode(ids[max(0, k - 48):k + 1]))
            buf_d.append(doc_id)
            buf_p.append(k)
            got += 1


pending = []
with torch.no_grad():
    for text in docs():
        if not text or len(text) < 800:
            continue
        di += 1
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=A.max_tok).input_ids
        if len(ids) < A.min_pos + 8:
            continue
        pending.append((di, ids))
        if len(pending) >= A.batch:
            run(pending)
            pending = []
            if len(buf_v) >= FLUSH:
                flush()
                el = time.time() - t0
                print("  %d/%d positions (%d docs) | %.0f pos/s | eta %.0f min"
                      % (got, A.n, di, got / el, (A.n - got) / max(got / el, 1) / 60), flush=True)
        if got >= A.n:
            break
if pending and got < A.n:
    run(pending)
if buf_v:
    flush()
w.close()
print("[hp] wrote %d positions from %d docs in %.0f min -> %s"
      % (got, di, (time.time() - t0) / 60, A.out), flush=True)
print("HARVEST_DONE", flush=True)
