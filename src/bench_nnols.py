#!/usr/bin/env python3
"""Where does the dictionary-miner inner loop spend its time?

384 rows took 33s at batch 192 -> 24h for 500k, which is far off the FLOP budget (the correlation
GEMM alone should run at hundreds of TFLOPS on a B200). Isolate the ops against a random bank of
the same shape so the 7-minute parquet load is not paid per experiment.
"""
import time
import torch

N, D = 1583873, 5120
dev = "cuda"
AJ = torch.randn(N, D, device=dev, dtype=torch.float16)
AJn = AJ.float().norm(dim=1).clamp(min=1e-6).half()
print("bank %.1f GB" % (AJ.numel() * 2 / 1e9), flush=True)


def t(fn, n=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


for B in (192, 1024, 4096):
    R = torch.randn(B, D, device=dev, dtype=torch.float16)
    c = torch.randint(0, N, (B,), device=dev)
    pen = torch.zeros(N, B, dtype=torch.float16, device=dev)
    print("\n--- B=%d ---" % B, flush=True)
    print("  gemm fp16 AJ@R.T        %.3f s" % t(lambda: AJ @ R.T))
    print("  + .float()              %.3f s" % t(lambda: (AJ @ R.T).float()))
    print("  + /AJn                  %.3f s" % t(lambda: (AJ @ R.T).float() / AJn.float().unsqueeze(1)))
    print("  + pen.float() add       %.3f s" % t(lambda: (AJ @ R.T).float() / AJn.float().unsqueeze(1) + pen.float()))
    print("  argmax(0) on [N,B]      %.3f s" % t(lambda: (AJ @ R.T).float().argmax(0)))
    sim = AJ @ AJ[c].T
    print("  gather AJ[c]            %.3f s" % t(lambda: AJ[c]))
    print("  sim gemm                %.3f s" % t(lambda: AJ @ AJ[c].T))
    print("  bool mask assign        %.3f s" % t(lambda: pen.masked_fill_(sim > 0.9, -1e4)))
    del R, pen, sim
    torch.cuda.empty_cache()

# fp16 accumulate vs fp32: is the GEMM using the fp32-accum path?
R = torch.randn(1024, D, device=dev, dtype=torch.float16)
flops = 2 * N * D * 1024
s = t(lambda: AJ @ R.T)
print("\nB=1024 gemm: %.3f s -> %.1f TFLOPS  (HBM-bound floor would be %.3f s at 6 TB/s)"
      % (s, flops / s / 1e12, AJ.numel() * 2 / 6e12), flush=True)
print("BENCH_DONE", flush=True)
