"""Write the modulation-lens target bank in the format maemm's rl_disagg.py memmaps.

    --bank-file vecs.f32   ->  np.memmap(dtype=float32, shape=(n_vecs, D_MODEL))

The bank holds RAW L42 activations, because that is the vector the policy gets INJECTED with
during generation (maemm's dirs serve double duty as the injected vector and the reward target).
The reward derives its own comparison space -- J, minus the activation-pool mean, unit-norm --
inside ARReward.target_space(). Writing pre-transformed vectors here would apply J twice AND
inject the wrong thing.

Also emits the sidecar the trainer's transcripts want: one line per bank row with the clause the
model was reading, so a rollout can be read next to its ground truth.
"""
import modal

app = modal.App("celeste-build-bank")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("numpy", "pyarrow"))


@app.function(image=img, volumes={"/vol": VOL}, cpu=8.0, memory=131072, timeout=7200)
def build(n: int = 500000, out_dir: str = "/vol/rl_bank", d_model: int = 5120,
          holdout: int = 2048):
    import json, os
    import numpy as np, pyarrow.parquet as pq
    os.makedirs(out_dir, exist_ok=True)
    pf = pq.ParquetFile("/vol/data/prose_L42_500k.parquet")
    total = min(n, pf.metadata.num_rows)
    # last `holdout` rows are reserved: the RL must never train on rows the readout eval uses
    n_tr = total - holdout
    print("[bank] %d rows -> %d train / %d holdout" % (total, n_tr, holdout), flush=True)

    for tag, lo, hi in (("vecs.f32", 0, n_tr), ("vecs_holdout.f32", n_tr, total)):
        path = os.path.join(out_dir, tag)
        mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(hi - lo, d_model))
        meta = open(os.path.join(out_dir, tag.replace(".f32", "_meta.jsonl")), "w")
        row0, w = 0, 0
        for bt in pf.iter_batches(batch_size=4096,
                                  columns=["activation_vector", "label", "ctx", "doc_id", "pos"]):
            b = bt.to_pydict()
            A = np.asarray(bt.column("activation_vector").flatten().to_numpy(zero_copy_only=False),
                           dtype="float32").reshape(-1, d_model)
            for j in range(A.shape[0]):
                r = row0 + j
                if lo <= r < hi:
                    mm[w] = A[j]
                    meta.write(json.dumps({"row": w, "src_row": r, "label": b["label"][j],
                                           "ctx": b["ctx"][j][-400:], "doc_id": b["doc_id"][j],
                                           "pos": b["pos"][j]}) + "\n")
                    w += 1
            row0 += A.shape[0]
            if row0 >= hi: break
        mm.flush(); del mm; meta.close()
        sz = os.path.getsize(path)
        assert sz == (hi - lo) * 4 * d_model, "size mismatch: %d" % sz
        print("[bank] %s: %d rows, %.2f GB, %d bytes/row" % (tag, hi - lo, sz / 1e9, 4 * d_model),
              flush=True)
        # the exact check rl_disagg does: n_vecs = filesize // (4 * D_MODEL)
        print("       rl_disagg would infer n_vecs = %d" % (sz // (4 * d_model)), flush=True)
    VOL.commit()
    print("BANK_DONE", flush=True)
