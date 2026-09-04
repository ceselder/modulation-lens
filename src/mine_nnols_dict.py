#!/usr/bin/env python3
"""Warm start from the THINKIES DICTIONARY instead of from the policy's own samples.

The select-mode miner sampled 4 candidates from the policy and picked the best composition, so its
bullets were near-rephrasings of each other -- the composition gain over the best single bullet was
only +0.012 cos, and NNLS zeroed three of four slots. This miner instead decomposes each activation
against the 1.58M-atom thinkies bank by greedy non-negative matching pursuit in J-space, and hands
the top-K atom LABELS over as the four bullets. Those atoms are complementary by construction: each
is selected to explain the residual the previous ones left.

Inspection of K=16 picks showed only ~2-4 atoms per activation are semantically coherent and the
rest are geometric filler with evenly-spread weights, so K is small here by design (default 4) --
the point is a format-plus-diversity warm start that RL can then improve, not a faithful
decomposition.

--dedupe-cos guards the failure mode this is meant to escape: the bank holds many near-duplicate
atoms, so plain greedy can pick four paraphrases of one concept (masking only the exact index it
already took). Atoms within that cosine of an already-chosen atom are excluded from later picks.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C

p = argparse.ArgumentParser()
p.add_argument("--data", default="/workspace/inv/data/prose_L42.parquet")
p.add_argument("--thinkies", default="/workspace/thinkies/v3/thinkies_v3-*.parquet")
p.add_argument("--out", default="/workspace/inv/data/nnols4.jsonl")
p.add_argument("--k", type=int, default=4)
p.add_argument("--n-acts", type=int, default=20000, help="how many FILTERED activations to mine")
p.add_argument("--n-pool", type=int, default=60000,
               help="must match the trainer's --n-pool: emitted `i` indexes the raw parquet order "
                    "over exactly this many rows, and the trainer asserts the range.")
p.add_argument("--min-words", type=int, default=4, help="match the trainer's --min-words")
p.add_argument("--max-tok", type=int, default=10,
               help="truncate each atom label to this many tokens, matching --bullet-max-tok. The "
                    "reported compose_cos is for the FULL atom vectors, so it is an upper bound on "
                    "what the truncated bullet text can read back to -- verify separately.")
p.add_argument("--dedupe-cos", type=float, default=0.9,
               help="exclude atoms this close to one already picked; 0 disables")
p.add_argument("--layer", type=int, default=42)
p.add_argument("--batch", type=int, default=192)
A = p.parse_args()
dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

# ---- dictionary ----
t0 = time.time()
J = C.load_jlens(A.layer, dev)
labs, vecs = [], []
for sh in sorted(glob.glob(A.thinkies)):
    for b in pq.ParquetFile(sh).iter_batches(batch_size=16384, columns=["label", "vector"]):
        l = b.column("label").to_pylist()
        labs += l
        vecs.append(np.asarray(b.column("vector").flatten().to_numpy(zero_copy_only=False),
                               dtype="float16").reshape(len(l), -1))
AJ = (torch.from_numpy(np.concatenate(vecs)).to(dev, torch.float16).float() @ J.T).half()
del vecs
AJn = AJ.float().norm(dim=1).clamp(min=1e-6).half()
NATOM = AJ.shape[0]
print("[i] %d atoms in J-space (%.0fs)" % (NATOM, time.time() - t0), flush=True)

# ---- activations: raw parquet order, trainer's min_words filter, trainer's centring ----
acc_v, acc_l = [], []
pf = pq.ParquetFile(A.data)
for b in pf.iter_batches(batch_size=4096, columns=["activation_vector", "label"]):
    d = b.to_pydict()
    acc_v.append(np.array(d["activation_vector"], dtype="float32"))
    acc_l.extend(d["label"])
    if sum(len(x) for x in acc_v) >= A.n_pool:
        break
V = np.concatenate(acc_v)
keep = [i for i, l in enumerate(acc_l) if len(l.split()) >= A.min_words][: A.n_pool]
# AMU is the trainer's target-side centre: the mean of the pool it will actually train on.
mu = torch.from_numpy(V[keep].mean(0)).to(dev)
mine = keep[: A.n_acts]
print("[i] pool %d raw rows -> %d after min_words=%d -> mining %d | max raw index %d"
      % (V.shape[0], len(keep), A.min_words, len(mine), max(mine)), flush=True)


def solve_nn(G, b, k):
    """Exact NNLS for k<=4 by active-set enumeration, batched. G [B,k,k], b [B,k] -> w [B,k]>=0.

    Enumerating all 2^k-1 supports and keeping the lowest-residual FEASIBLE one is exact for small
    k, and it is what the trainer's reward uses -- an lstsq-then-clamp would report a different
    number than RL will score.
    """
    B = G.shape[0]
    best_w = torch.zeros(B, k, dtype=torch.float64, device=G.device)
    best_r = torch.full((B,), float("inf"), dtype=torch.float64, device=G.device)
    for mask in range(1, 1 << k):
        s = [j for j in range(k) if mask >> j & 1]
        idx = torch.tensor(s, device=G.device)
        Gs = G[:, idx][:, :, idx]
        bs = b[:, idx]
        try:
            ws = torch.linalg.solve(Gs + 1e-8 * torch.eye(len(s), dtype=G.dtype, device=G.device),
                                    bs.unsqueeze(2)).squeeze(2)
        except Exception:
            continue
        feas = (ws >= -1e-9).all(1) & torch.isfinite(ws).all(1)
        # residual^2 = ||t||^2 - 2 w.b + w'Gw, and ||t||^2 is constant across supports
        r = (-2 * (ws * bs).sum(1) + (ws.unsqueeze(1) @ Gs @ ws.unsqueeze(2)).squeeze()).view(B)
        take = feas & (r < best_r)
        best_r = torch.where(take, r, best_r)
        w_full = torch.zeros(B, k, dtype=torch.float64, device=G.device)
        w_full[:, idx] = ws.clamp(min=0)
        best_w = torch.where(take.unsqueeze(1), w_full, best_w)
    return best_w


nkept, nrow = 0, 0
os.makedirs(os.path.dirname(A.out), exist_ok=True)
f = open(A.out, "w")
gain, comp, sing, nnz = [], [], [], []
for b0 in range(0, len(mine), A.batch):
    bidx = mine[b0: b0 + A.batch]
    T = torch.from_numpy(V[bidx]).to(dev) - mu
    T = (T @ J.T)
    T = T / T.norm(dim=1, keepdim=True).clamp(min=1e-8)          # [B,5120] unit targets
    B = T.shape[0]
    pen = torch.zeros(NATOM, B, dtype=torch.float16, device=dev)  # additive exclusion mask
    chosen = torch.zeros(B, A.k, dtype=torch.long, device=dev)
    hist = []
    resid = T.clone()
    for step in range(A.k):
        corr = (AJ @ resid.half().T).float() / AJn.float().unsqueeze(1)     # [N,B]
        corr += pen.float()
        c = corr.argmax(0)
        chosen[:, step] = c
        if A.dedupe_cos > 0:
            sim = (AJ @ AJ[c].T).float() / (AJn.float().unsqueeze(1) * AJn[c].float().unsqueeze(0))
            pen[sim > A.dedupe_cos] = -1e4
        else:
            pen[c, torch.arange(B, device=dev)] = -1e4
        del corr
        S = AJ[chosen[:, : step + 1].reshape(-1)].float().view(B, step + 1, -1)   # [B,k',5120]
        G = (S @ S.transpose(1, 2)).double()
        rhs = (S @ T.unsqueeze(2)).squeeze(2).double()
        w = solve_nn(G, rhs, step + 1)
        rec = (w.float().unsqueeze(1) @ S).squeeze(1)                             # [B,5120]
        hist.append(((rec * T).sum(1) / rec.norm(dim=1).clamp(min=1e-8)).cpu())
        resid = T - rec
        del S, G, rhs, rec
    W = w.cpu().numpy()
    CH = chosen.cpu().numpy()
    for r in range(B):
        order = np.argsort(-W[r])
        full = [labs[int(CH[r, j])] for j in order]
        bull = [(tok.decode(tok(x, add_special_tokens=False).input_ids[: A.max_tok]).strip()
                 if len(tok(x, add_special_tokens=False).input_ids) > A.max_tok else x) or "the"
                for x in full]
        f.write(json.dumps({"i": int(bidx[r]), "label": acc_l[bidx[r]], "bullets": bull,
                            "atom_labels_full": full,
                            "weights": [round(float(W[r, j]), 5) for j in order],
                            "compose_cos": round(float(hist[-1][r]), 4),
                            "single_cos": round(float(hist[0][r]), 4)}) + "\n")
        comp.append(float(hist[-1][r]))
        sing.append(float(hist[0][r]))
        gain.append(float(hist[-1][r] - hist[0][r]))
        nnz.append(int((W[r] > 1e-6).sum()))
        nkept += 1
    del pen
    torch.cuda.empty_cache()
    if b0 % (A.batch * 5) == 0:
        print("[mine] %d/%d | single %.4f -> compose %.4f (+%.4f) | mean nnz %.2f | %.0fs"
              % (nkept, len(mine), float(np.mean(sing)), float(np.mean(comp)),
                 float(np.mean(gain)), float(np.mean(nnz)), time.time() - t0), flush=True)
f.close()
print("[done] %d rows -> %s | single %.4f compose %.4f gain %+.4f | nnz %.2f/%d | dedupe %.2f"
      % (nkept, A.out, float(np.mean(sing)), float(np.mean(comp)), float(np.mean(gain)),
         float(np.mean(nnz)), A.k, A.dedupe_cos), flush=True)
print("MINE_NNOLS_DONE", flush=True)
