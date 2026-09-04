"""Copy the thinkies-v3 dictionary onto the Modal volume.

1,583,873 rows x (label, reliability, n_words, vector[5120] fp16) in 7 shards, 15.6 GB. Pulled
here rather than locally because the AR trainer runs on Modal and the box that held the original
/workspace/thinkies/v3/ copy is down -- the Hub is now the only surviving source.

Asserts the schema and row count, because a silently-truncated download would show up much later as
an AR that simply fits worse.
"""
import os
import shutil

import modal

app = modal.App("celeste-fetch-thinkies")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("huggingface_hub[hf_transfer]", "pyarrow", "numpy")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))

FILES = ([f"thinkies_v3-{i:05d}-of-00007.parquet" for i in range(7)]
         + ["thinkies_v3_ref_mean.npy", "thinkies_v3_preview.parquet", "README.md"])


@app.function(image=img, volumes={"/vol": VOL}, timeout=7200,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def fetch():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    out = "/vol/thinkies_v3"
    os.makedirs(out, exist_ok=True)
    got = []
    for f in FILES:
        dst = os.path.join(out, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            print("[skip] %s (%.0f MB present)" % (f, os.path.getsize(dst) / 1e6), flush=True)
            got.append(dst); continue
        try:
            # download STRAIGHT into the volume: /tmp and /vol are different filesystems, so an
            # os.replace across them raises EXDEV ("Invalid cross-device link").
            p = hf_hub_download("ceselder/thinkies-v3", f, repo_type="dataset",
                                token=os.environ["HF_TOKEN"], local_dir=out + "/_dl")
        except Exception as e:
            print("[miss] %s (%s)" % (f, type(e).__name__), flush=True); continue
        shutil.move(p, dst)
        print("[got ] %s  %.0f MB" % (f, os.path.getsize(dst) / 1e6), flush=True)
        got.append(dst)
        VOL.commit()
    shards = sorted(x for x in got if "-of-00007" in x)
    total = 0
    for s in shards:
        pf = pq.ParquetFile(s)
        names = [f.name for f in pf.schema_arrow]
        assert {"label", "vector", "reliability"} <= set(names), (s, names)
        v = pf.schema_arrow.field("vector").type
        assert v.list_size == 5120, (s, v)
        total += pf.metadata.num_rows
    print("[ok] %d shards, %d rows, vector[5120] fp16" % (len(shards), total), flush=True)
    assert total == 1583873, "expected 1,583,873 atoms, got %d -- truncated download" % total
    VOL.commit()
    print("THINKIES_READY", flush=True)
    return total
