import modal
app = modal.App("modlens-verify-dict")
vol = modal.Volume.from_name("celeste-modlens-vol")
image = modal.Image.debian_slim(python_version="3.12").pip_install("pyarrow", "numpy")

@app.function(image=image, volumes={"/vol": vol}, timeout=1800, memory=32768)
def run():
    import glob, pyarrow.parquet as pq
    for tier in ("all", "f065"):
        mp = "/vol/dict_5m/meta_%s.parquet" % tier
        m = pq.ParquetFile(mp)
        nm = m.metadata.num_rows
        shards = sorted(glob.glob("/vol/dict_5m/vec_%s_*.parquet" % tier))
        nv = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
        print("[{}] meta rows {:,} | {} vec shards, {:,} rows | MATCH {}".format(
            tier, nm, len(shards), nv, nm == nv), flush=True)
        print("      meta columns:", [f.name for f in m.schema_arrow][:14], flush=True)
    return True

@app.local_entrypoint()
def main():
    run.remote()
