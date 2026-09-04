"""Are adjacent rows of the RL bank correlated?

The contrastive reward takes its negative from the NEXT GROUP (stride = group_size). That is only a
valid negative if row i and row i+stride are unrelated activations. If the bank is ordered by
document -- many positions from one doc in a row -- then the 'negative' is a near-duplicate of the
positive, the contrast collapses to ~0, and (worse) every GRPO batch is built from correlated
targets. Measured against the training reward reading 0.002 where the permutation control predicted
0.21.
"""
import json, os, modal
app = modal.App("modlens-diag-bank-order")
vol = modal.Volume.from_name("celeste-modlens-vol")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")
         .env({"HF_HOME": "/vol/.hf_home"})
         .add_local_file("rl/diag_bank_order.py", "/root/d.py"))

@app.function(image=image, volumes={"/vol": vol}, gpu="A10G", timeout=1800, memory=32768)
def run(n: int = 40000):
    import numpy as np, torch
    dev = "cuda"
    H = np.fromfile("/vol/rl_bank/vecs.f32", dtype="float32").reshape(-1, 5120)[:n]
    T = torch.nn.functional.normalize(torch.from_numpy(H).to(dev).float(), dim=-1)
    out = {}
    for stride in (1, 2, 4, 16, 64, 256, 4096):
        a = T[:-stride]; b = T[stride:]
        c = (a * b).sum(-1)
        out["stride_%d" % stride] = [float(c.mean()), float(c.std())]
        print("  cos(row i, row i+%-5d) = %.4f (sd %.4f)" % (stride, c.mean(), c.std()), flush=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    p = torch.randperm(T.shape[0], generator=g).to(dev)
    c = (T * T[p]).sum(-1)
    out["random_pairs"] = [float(c.mean()), float(c.std())]
    print("  cos(random pairs)            = %.4f (sd %.4f)" % (c.mean(), c.std()), flush=True)
    # also: how many rows share a doc, from the meta
    try:
        docs = [json.loads(l).get("doc_id") for l in open("/vol/rl_bank/vecs_meta.jsonl")][:n]
        runs, cur, best = 1, docs[0], 1
        for d in docs[1:]:
            runs = runs + 1 if d == cur else 1
            cur = d; best = max(best, runs)
        from collections import Counter
        cc = Counter(docs)
        print("  meta: %d rows, %d distinct doc_ids, longest consecutive same-doc run = %d, "
              "max rows per doc = %d" % (len(docs), len(cc), best, max(cc.values())), flush=True)
        out["meta"] = {"rows": len(docs), "distinct_docs": len(cc), "longest_run": best,
                       "max_rows_per_doc": max(cc.values())}
    except Exception as e:
        print("  [meta] %s" % str(e)[:120], flush=True)
    os.makedirs("/vol/diag", exist_ok=True)
    json.dump(out, open("/vol/diag/bank_order.json", "w"), indent=1)
    vol.commit()
    return out

@app.local_entrypoint()
def main(n: int = 40000):
    print(json.dumps(run.remote(n=n), indent=1))
