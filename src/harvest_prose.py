#!/usr/bin/env python3
"""
Harvest L42 activations from the pretrain corpus, labelled with the VERBATIM PRECEDING TEXT.

Replaces two things lost with the old box in a single pass: the activation pool (av_L42_150k) and
the SFT labels. The old labels were the nearest dictionary atom, which needed the 16 harvest
templates to be meaningful -- and those template strings are gone, so the published atoms can no
longer be reproduced in the same geometry. The preceding text needs no dictionary at all: it is
what the model was literally reading when the state occurred, so its provenance is exact and it is
fluent English by construction.

Read CHAT-NATIVELY (passage inside a user turn). Measured on the old box: raw-vs-chat reads differ
by whitened cosine 0.75 averaged over positions and as low as 0.06 at the worst, so a raw read
describes a state the model never occupies in deployment.

Emits, per position:
  activation_vector  L42 at that position
  label              the preceding span, <= --label-tok tokens, ending AT the position
  ctx                a longer window for eyeballing
  doc_id, pos        provenance
"""
import argparse, glob, json, os, random
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="", help="parquet with a 'text' column; blank = HF fineweb stream")
ap.add_argument("--layer", type=int, default=42)
ap.add_argument("--n", type=int, default=60000, help="positions to harvest")
ap.add_argument("--per-doc", type=int, default=6, help="positions sampled per document")
ap.add_argument("--max-tok", type=int, default=256)
ap.add_argument("--min-pos", type=int, default=24, help="skip early positions: too little context")
ap.add_argument("--label-tok", type=int, default=16, help="preceding tokens used as the label")
ap.add_argument("--at-boundary", type=int, default=1,
                help="Sample only positions whose token is a clause/sentence delimiter. Measured on "
                     "uniformly-sampled positions, the preceding-text label separates matched from "
                     "mismatched by 14x on average but wins outright at only 7/12 positions -- and "
                     "the failures all end mid-clause ('...members argued that', -0.017) while the "
                     "wins are complete phrases ('...movements she had never managed to finish.', "
                     "0.367). A dangling span is an incoherent thing to hold in mind, which is what "
                     "the reward literally asks for.")
ap.add_argument("--out", default="/workspace/inv/data/prose_L42.parquet")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
random.seed(A.seed)
BASE = "Qwen/Qwen3.6-27B"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
G = {}
model.model.layers[A.layer].register_forward_hook(
    lambda m, i, o: G.__setitem__("h", o[0] if isinstance(o, tuple) else o))

# chat wrapper, split on a sentinel so the passage's own tokens are untouched and position i of the
# passage stays position i
_r = tok.apply_chat_template([{"role": "user", "content": "XSLOT"}], tokenize=False,
                             add_generation_prompt=True, enable_thinking=False)
_a, _b = _r.split("XSLOT")
CPRE = tok(_a, add_special_tokens=False).input_ids
CPOST = tok(_b, add_special_tokens=False).input_ids
print("[hp] chat wrapper: %d + passage + %d tokens" % (len(CPRE), len(CPOST)), flush=True)


DELIM = set()
for _s in ". , ! ? ; : ... \u2014".split():
    for _c in (_s, " " + _s):
        _i = tok(_c, add_special_tokens=False).input_ids
        if len(_i) == 1:
            DELIM.add(_i[0])
print("[hp] %d delimiter token ids" % len(DELIM), flush=True)


def docs():
    if A.corpus:
        for f in sorted(glob.glob(A.corpus)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["text"]):
                for t in b.to_pydict()["text"]:
                    yield t
    else:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train",
                          streaming=True, token=os.environ.get("HF_TOKEN"))
        for r in ds:
            yield r["text"]


D = 5120
schema = pa.schema([("activation_vector", pa.list_(pa.float32(), D)),
                    ("label", pa.string()), ("ctx", pa.string()),
                    ("doc_id", pa.int64()), ("pos", pa.int64())])
w = pq.ParquetWriter(A.out, schema)
buf_v, buf_l, buf_c, buf_d, buf_p = [], [], [], [], []
got, di = 0, 0
FLUSH = 4096
with torch.no_grad():
    for text in docs():
        if not text or len(text) < 800:
            continue
        di += 1
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=A.max_tok).input_ids
        if len(ids) < A.min_pos + 8:
            continue
        model(input_ids=torch.tensor([CPRE + ids + CPOST], device="cuda"))
        H = G["h"].float()[0][len(CPRE):len(CPRE) + len(ids)]
        cand = list(range(A.min_pos, len(ids)))
        if A.at_boundary:
            cand = [k for k in cand if ids[k] in DELIM]
        random.shuffle(cand)
        for k in cand[: A.per_doc]:
            buf_v.append(H[k].cpu().numpy())
            # label ENDS at the position: the text the model had just read
            # start the label at the PREVIOUS boundary when one is within reach, so the span is
            # a self-contained clause rather than a window that begins mid-phrase
            lo = max(0, k - A.label_tok + 1)
            for j in range(k - 1, lo - 1, -1):
                if ids[j] in DELIM:
                    lo = j + 1
                    break
            buf_l.append(tok.decode(ids[lo:k + 1]).strip())
            buf_c.append(tok.decode(ids[max(0, k - 48):k + 1]))
            buf_d.append(di)
            buf_p.append(k)
            got += 1
        if len(buf_v) >= FLUSH:
            w.write_table(pa.table({
                "activation_vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.stack(buf_v).reshape(-1), type=pa.float32()), D),
                "label": pa.array(buf_l), "ctx": pa.array(buf_c),
                "doc_id": pa.array(buf_d, type=pa.int64()),
                "pos": pa.array(buf_p, type=pa.int64())}, schema=schema))
            buf_v, buf_l, buf_c, buf_d, buf_p = [], [], [], [], []
            print("  %d/%d positions (%d docs)" % (got, A.n, di), flush=True)
        if got >= A.n:
            break
if buf_v:
    w.write_table(pa.table({
        "activation_vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.stack(buf_v).reshape(-1), type=pa.float32()), D),
        "label": pa.array(buf_l), "ctx": pa.array(buf_c),
        "doc_id": pa.array(buf_d, type=pa.int64()),
        "pos": pa.array(buf_p, type=pa.int64())}, schema=schema))
w.close()
pf = pq.ParquetFile(A.out)
print("\nwrote %s: %d rows from %d docs" % (A.out, pf.metadata.num_rows, di), flush=True)
s = pf.read_row_group(0).slice(0, 6).to_pylist()
for r in s:
    print("   pos %-4d label %r" % (r["pos"], r["label"][:70]))
print("HARVEST_PROSE_DONE", flush=True)
