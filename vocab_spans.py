"""The full vocabulary as 1-token dictionary atoms.

For length 1 there is no reason to MINE: the set of all single tokens is finite (248,320 for
Qwen3.6-27B) and cheap to measure -- 248k x 16 cells = 4M cell-reads, about 40 minutes on 8 GPUs.
So the 1-token layer of the dictionary can be COMPLETE rather than sampled, which is a property no
mined layer can have.

Filtered to tokens that are usable as an atom label:
  * decodes to a non-empty printable string
  * not a special/added token (<|im_start|> etc. are control structure, not concepts)
  * not pure whitespace
Byte-fallback fragments ARE kept: they are real directions the model uses, and reliability will
sort out whether they steer anything.
"""
import os

import modal

app = modal.App("celeste-vocab-spans")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("transformers==5.5.4", "tokenizers", "pyarrow", "numpy",
                    "huggingface_hub[hf_transfer]")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false"}))


@app.function(image=img, volumes={"/vol": VOL}, cpu=4.0, timeout=3600)
def build():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    n = len(tok)
    special = set(tok.all_special_ids or [])
    try:
        special |= {i for i in tok.get_added_vocab().values()}
    except Exception:
        pass
    keep, kept_ids, dropped = [], [], {"special": 0, "empty": 0, "space": 0, "reencode": 0}
    for i in range(n):
        if i in special:
            dropped["special"] += 1; continue
        s = tok.decode([i])
        if not s:
            dropped["empty"] += 1; continue
        if not s.strip():
            dropped["space"] += 1; continue
        # the atom label must round-trip to the SAME single token, or the read is of something else
        if tok(s, add_special_tokens=False).input_ids != [i]:
            dropped["reencode"] += 1; continue
        keep.append(s); kept_ids.append(i)
    print("[vocab] %d of %d tokens kept | dropped %s" % (len(keep), n, dropped), flush=True)
    os.makedirs("/vol/spans_vocab", exist_ok=True)
    pq.write_table(pa.table({
        "span": pa.array(keep, pa.string()),
        "n_tokens": pa.array([1] * len(keep), pa.int16()),
        "token_id": pa.array(kept_ids, pa.int32()),
        "domain": pa.array(["__vocab__"] * len(keep), pa.string()),
        "source": pa.array(["qwen3.6-27b-vocab"] * len(keep), pa.string()),
    }), "/vol/spans_vocab/vocab-00000.parquet", compression="zstd")
    VOL.commit()
    print("[vocab] examples:", [repr(x) for x in keep[:6]], flush=True)
    print("VOCAB_DONE %d" % len(keep), flush=True)
    return len(keep)
