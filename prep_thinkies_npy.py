"""Convert thinkies-v3 to .npy so the playground stops re-parsing 15.6 GB of parquet on every
cold start. Same trick already used for the FineFineWeb bank: parquet is a great archive format and
a terrible load-on-startup format."""
import os
import modal
app = modal.App("celeste-prep-thinkies-npy")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = modal.Image.debian_slim(python_version="3.12").pip_install("pyarrow", "numpy")

@app.function(image=img, volumes={"/vol": VOL}, cpu=8.0, memory=196608, timeout=7200)
def run():
    import glob, json
    import numpy as np, pyarrow.parquet as pq
    if os.path.exists("/vol/thinkies_npy/vectors.npy"):
        print("already converted"); return 0
    labs, vecs, rels = [], [], []
    for sh in sorted(glob.glob("/vol/thinkies_v3/thinkies_v3-*-of-00007.parquet")):
        for b in pq.ParquetFile(sh).iter_batches(batch_size=16384,
                                                 columns=["label", "vector", "reliability"]):
            labs += b.column("label").to_pylist()
            rels.append(np.asarray(b.column("reliability").to_numpy(zero_copy_only=False), dtype="float32"))
            vecs.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                                   dtype="float16").reshape(-1, 5120))
        print("  %s atoms" % "{:,}".format(sum(v.shape[0] for v in vecs)), flush=True)
    os.makedirs("/vol/thinkies_npy", exist_ok=True)
    V = np.concatenate(vecs); np.save("/vol/thinkies_npy/vectors.npy", V)
    np.save("/vol/thinkies_npy/rel.npy", np.concatenate(rels))
    json.dump(labs, open("/vol/thinkies_npy/labels.json", "w"))
    VOL.commit()
    print("THINKIES_NPY_DONE %d atoms, %.1f GB" % (len(labs), V.nbytes/1e9), flush=True)
    return len(labs)
