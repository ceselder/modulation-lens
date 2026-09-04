"""Honest MFU meter. Numerator counts only REAL (non-pad) tokens → padding lowers MFU, as it should.

MFU = achieved_model_FLOPs / (wall_time * ROOFLINE). ROOFLINE is the *measured* B300 bf16 matmul
ceiling (~1500 TFLOP/s at the power-capped 1650 MHz with torch-cu130), not a marketing spec — a
pure matmul cannot beat it, so this is the honest denominator for "how close to the metal are we".
"""
ROOFLINE_TFLOPS = 1500.0  # measured: `n=16384 bf16 a@b` on an idle B300 (override via env if kernels improve)


def flops_per_token(n_params: int, fwd_bwd: bool, layer_frac: float = 1.0) -> float:
    """6ND (fwd+bwd) or 2ND (fwd only). layer_frac<1 for early-exit forwards (cache_resids @ L27)."""
    return (6.0 if fwd_bwd else 2.0) * n_params * layer_frac


def mfu(n_real_tokens: int, seconds: float, n_params: int, fwd_bwd: bool = True,
        layer_frac: float = 1.0, roofline_tflops: float = ROOFLINE_TFLOPS) -> tuple[float, float]:
    """Returns (achieved_TFLOP_s, mfu_fraction) vs the measured roofline."""
    tflops = flops_per_token(n_params, fwd_bwd, layer_frac) * n_real_tokens / seconds / 1e12
    return tflops, tflops / roofline_tflops
