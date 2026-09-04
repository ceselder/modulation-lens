"""CPU unit tests for the ScaleRL variant of rl/rl_disagg.py (no GPU, no vLLM, no HF model): flag bundle resolution,
CISPO vs PPO per-token loss/gradients, token/seq/prompt loss aggregation (+ effective-batch mask), batch-level
advantage normalization + zero-variance filter, No-Positive-Resampling bookkeeping + drop-aware direction sampling,
and the fp32 lm_head hook.

    python -m pytest rl/test_rl_disagg_scalerl.py -q      (or: python rl/test_rl_disagg_scalerl.py)
"""
import importlib.util
import json
import os
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rl_disagg", os.path.join(_HERE, "rl_disagg.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

_BASE = ["--role", "launch", "--n-rollout", "2", "--n-trainer", "6", "--groups-per-step", "256", "--group-size", "8"]


# ------------------------------------------------------------------ flags
def test_legacy_defaults_are_the_pre_scalerl_behaviour():
    a = D.parse_args(_BASE)
    assert (a.recipe, a.loss, a.loss_agg, a.adv_mode, a.zero_var_filter, a.npr_threshold, a.max_lag, a.fp32_head, a.length_control) == \
        ("", "ppo", "token", None, False, 0.0, 1, False, "penalty")
    assert a.max_queue_blocks == a.blocks_per_step == 2 and a.max_num_seqs == 1024 and a.len_penalty_start == 64


def test_scalerl_bundle_and_explicit_override():
    a = D.parse_args(_BASE + ["--recipe", "scalerl"])
    assert (a.loss, a.cispo_eps_max, a.loss_agg, a.adv_mode, a.zero_var_filter, a.npr_threshold, a.max_lag, a.fp32_head) == \
        ("cispo", 5.0, "prompt", "batch", True, 0.9, 8, True)
    assert a.length_control == "penalty" and a.len_penalty_start == 64          # the bundle does NOT touch the hinge LP
    assert a.max_queue_blocks == 8 * a.blocks_per_step == 16                    # PipelineRL-8 queue depth
    # explicit flags win over the bundle, in either order
    a = D.parse_args(_BASE + ["--loss", "ppo", "--recipe", "scalerl", "--max-lag", "3", "--no-fp32-head", "--adv-mode", "group",
                              "--loss-agg", "seq", "--no-zero-var-filter", "--npr-threshold", "0"])
    assert (a.loss, a.max_lag, a.fp32_head, a.adv_mode, a.loss_agg, a.zero_var_filter, a.npr_threshold) == ("ppo", 3, False, "group", "seq", False, 0.0)
    assert a.max_queue_blocks == 6
    a = D.parse_args(_BASE + ["--max-lag", "8", "--max-queue-blocks", "5"])   # explicit queue cap beats --max-lag
    assert a.max_queue_blocks == 5
    a = D.parse_args(_BASE + ["--length-control", "interrupt"])
    assert a.len_penalty_start is None
    try:
        D.parse_args(_BASE + ["--max-lag", "4", "--drop-stale"])
        assert False, "drop-stale must be rejected with a lag > 1"
    except AssertionError as e:
        assert "FIFO" in str(e)


# ------------------------------------------------------------------ loss
def _grad(loss_name, rho, A, eps=0.2, tis=2.0, eps_max=5.0):
    old = torch.zeros(1, len(rho))
    new = torch.log(torch.tensor(rho, dtype=torch.float32)).view(1, -1).clone().requires_grad_(True)
    Aa = torch.tensor([[A]], dtype=torch.float32)
    loss_tok, ratio, rho_raw = D.pg_token_loss(new, old, Aa, loss_name, eps, tis, eps_max)
    loss_tok.sum().backward()
    return new.grad.view(-1), ratio.detach().view(-1), rho_raw.view(-1)


def test_cispo_weights_vs_ppo_on_synthetic_ratios():
    rho = [0.5, 0.9, 1.0, 1.1, 1.5, 3.0, 10.0]
    # CISPO: d loss / d logp = -min(rho, eps_max) * A for EVERY token (no dead zone), weight truncated at eps_max
    g, w, rr = _grad("cispo", rho, A=1.0)
    assert torch.allclose(g, -torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 3.0, 5.0]))
    assert torch.allclose(w, torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 3.0, 5.0])) and torch.allclose(rr, torch.tensor(rho))
    g_neg, _, _ = _grad("cispo", rho, A=-1.0)
    assert torch.allclose(g_neg, -g)                       # linear in A, still no lower clip
    # PPO (rl.py): tokens outside the trust region in the direction of the advantage get ZERO gradient; the TIS cap (2.0)
    # bounds the ratio itself. A > 0: rho > 1.2 clipped -> 0 grad; A < 0: rho < 0.8 clipped -> 0 grad.
    g_ppo, w_ppo, _ = _grad("ppo", rho, A=1.0)
    assert torch.allclose(g_ppo, -torch.tensor([0.5, 0.9, 1.0, 1.1, 0.0, 0.0, 0.0]))
    assert torch.allclose(w_ppo, torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 2.0]))
    g_ppo_neg, _, _ = _grad("ppo", rho, A=-1.0)
    # ... and the TIS cap is a clamp: tokens with rho >= tis_cap have ZERO gradient for either sign of A (CISPO keeps
    # a truncated eps_max * A gradient there -- the behavioural difference between the two losses on stale tokens)
    assert torch.allclose(g_ppo_neg, torch.tensor([0.0, 0.9, 1.0, 1.1, 1.5, 0.0, 0.0]))
    # the PPO branch is bit-identical to the original rl_disagg expression
    new = torch.randn(4, 7); old = torch.randn(4, 7); A = torch.randn(4, 1)
    ratio = torch.exp(new - old).clamp(max=2.0)
    ref = -torch.minimum(ratio * A, ratio.clamp(0.8, 1.2) * A)
    lt, _, _ = D.pg_token_loss(new, old, A, "ppo", 0.2, 2.0, 5.0)
    assert torch.equal(lt, ref)


def test_loss_weights_token_seq_prompt_and_effective_batch():
    G, T = 4, 6
    lens = torch.tensor([3, 6, 1, 2, 4, 4, 5, 6])             # 2 groups x 4 rollouts
    gen_mask = torch.arange(T)[None, :] < lens[:, None]
    n = len(lens)
    # legacy paths are the original expressions bit for bit
    w_tok, s_tok = D.loss_weights(gen_mask, "token", G)
    assert torch.equal(w_tok, gen_mask.float() / int(gen_mask.sum())) and s_tok == float(gen_mask.sum())
    w_seq, s_seq = D.loss_weights(gen_mask, "seq", G)
    assert torch.equal(w_seq, gen_mask.float() / gen_mask.sum(1, keepdim=True).clamp(min=1).float() / n) and s_seq == float(n)
    # prompt: every group weighs 1/n_groups, uniform over ITS tokens
    w_pr, s_pr = D.loss_weights(gen_mask, "prompt", G)
    assert s_pr == 2.0 and abs(w_pr.sum().item() - 1.0) < 1e-6
    gsum = w_pr.view(2, G, T).sum((1, 2))
    assert torch.allclose(gsum, torch.tensor([0.5, 0.5]))
    assert torch.allclose(w_pr[0, :3], torch.full((3,), 1 / 12 / 2)) and torch.allclose(w_pr[7, :6], torch.full((6,), 1 / 19 / 2))
    assert w_pr[2, 1] == 0 and w_pr[0, 3] == 0                # padding stays inert
    # token/seq/prompt differ from each other (the whole point)
    assert not torch.allclose(w_pr, w_tok) and not torch.allclose(w_pr, w_seq) and not torch.allclose(w_seq, w_tok)
    # effective batch: dropping group 0 -> its rows weigh 0 and the denominators only count group 1
    keep = torch.tensor([False] * 4 + [True] * 4)
    for agg, exp_sync in (("prompt", 1.0), ("seq", 4.0), ("token", 19.0)):
        w, s = D.loss_weights(gen_mask, agg, G, keep)
        assert s == exp_sync and w[:4].abs().sum() == 0 and abs(w.sum().item() - 1.0) < 1e-6, agg
    w, _ = D.loss_weights(gen_mask, "prompt", G, keep)
    assert torch.allclose(w[4:][gen_mask[4:]], torch.full((19,), 1 / 19))
    w_all_keep, _ = D.loss_weights(gen_mask, "prompt", G, torch.ones(n, dtype=torch.bool))
    assert torch.allclose(w_all_keep, w_pr)


# ------------------------------------------------------------------ advantages
def test_batch_normalization_and_zero_variance_filter():
    G = 4
    r = torch.tensor([1.0, 1.0, 1.0, 1.0,          # zero-variance group
                      0.0, 1.0, 2.0, 3.0,
                      -1.0, 1.0, -1.0, 1.0])
    adv, keep = D.compute_advantages_disagg(r, 3, G, "batch", 1e-6, zero_var_filter=True)
    assert keep.tolist() == [False] * 4 + [True] * 8
    assert torch.all(adv[:4] == 0)                                       # zero-variance group -> zero advantage
    surv = adv[4:].double()
    assert abs(surv.mean().item()) < 1e-6 and abs(surv.pow(2).mean().sqrt().item() - 1.0) < 1e-4   # ONE batch std, not per group
    centered = torch.tensor([-1.5, -0.5, 0.5, 1.5, -1.0, 1.0, -1.0, 1.0])
    std = centered.double().pow(2).mean().sqrt().item()
    assert torch.allclose(adv[4:], centered / (std + 1e-6), atol=1e-5)
    # group mode = GRPO per-group std; none = Dr. GRPO centering; filter off -> keep None (legacy weights)
    adv_g, keep_g = D.compute_advantages_disagg(r, 3, G, "group")
    assert keep_g is None and torch.allclose(adv_g[4:8], centered[:4] / (torch.tensor([0.0, 1.0, 2.0, 3.0]).std() + 1e-6))
    adv_n, _ = D.compute_advantages_disagg(r, 3, G, "none")
    assert torch.allclose(adv_n[4:], centered)
    # the epsilon is honoured: a near-degenerate group is dropped with a looser threshold
    r2 = r.clone(); r2[:4] = torch.tensor([1.0, 1.0 + 1e-4, 1.0, 1.0])
    _, k_tight = D.compute_advantages_disagg(r2, 3, G, "batch", 1e-6, True)
    _, k_loose = D.compute_advantages_disagg(r2, 3, G, "batch", 1e-3, True)
    assert k_tight[0] and not k_loose[0]
    # matches rl.py's compute_advantages (the ScaleRL 'batch' mode rl_disagg used to call) when importable
    try:
        sp = importlib.util.spec_from_file_location("rl_hf", os.path.join(_HERE, "rl.py"))
        R = importlib.util.module_from_spec(sp); sp.loader.exec_module(R)
    except Exception:  # noqa — heavy deps missing on this box: the formula above is the spec
        return
    for mode in ("none", "group", "batch"):
        assert torch.allclose(R.compute_advantages(r, 3, G, mode), D.compute_advantages_disagg(r, 3, G, mode)[0]), mode


# ------------------------------------------------------------------ No-Positive-Resampling
def test_npr_tracker_bookkeeping_publish_and_sampling():
    G = 8
    npr = D.NPRTracker(threshold=0.9, pass_cos=0.7)
    idx = np.array([5, 6, 7, -1])
    cos = np.concatenate([np.full(8, 0.9),                        # dir 5: 8/8 positives -> dropped now
                          np.array([0.9] * 7 + [0.1]),            # dir 6: 7/8 = 0.875 < 0.9 -> not yet
                          np.full(8, 0.2),                        # dir 7: 0/8
                          np.full(8, 0.99)])                      # random direction: ignored
    st = npr.update(idx, cos, G)
    assert npr.dropped == {5} and st["scalerl/npr_new_dropped"] == 1 and st["scalerl/npr_dropped_total"] == 1
    assert st["scalerl/npr_directions_seen"] == 3 and abs(st["scalerl/npr_batch_flagged_frac"] - 0.25) < 1e-9
    assert abs(st["scalerl/npr_pass_frac"] - (8 + 7 + 0 + 8) / 32) < 1e-9
    # history is cumulative: dir 6 revisited with 8/8 -> 15/16 = 0.9375 >= 0.9 -> dropped
    st = npr.update(np.array([6]), np.full(8, 0.95), G)
    assert npr.dropped == {5, 6} and npr.hist[6] == [15, 16]
    w = tempfile.mkdtemp()
    path = f"{w}/npr/dropped.json"
    assert npr.publish(path) and json.load(open(path)) == [5, 6]
    assert not npr.publish(path)                                  # unchanged -> not rewritten
    rd = D._NPRDropList(path)
    assert rd.refresh() == {5, 6}
    assert D._NPRDropList(f"{w}/nope.json").refresh() == set()
    # sampling: without a drop set the ORIGINAL draw (same rng stream); with one, never a dropped row, distinct, sorted
    lo, hi, Bb = 3, 50, 10
    for seed in range(5):
        a1 = D._sample_block_idx(np.random.default_rng(seed), lo, hi, Bb, None)
        rng = np.random.default_rng(seed)
        a2 = lo + np.sort(rng.choice(hi - lo, size=Bb, replace=False))
        assert np.array_equal(a1, a2)
    dropped = set(range(3, 40))                                   # only rows 40..49 remain -> must return exactly those
    out = D._sample_block_idx(np.random.default_rng(0), lo, hi, Bb, dropped)
    assert out.tolist() == list(range(40, 50))
    dropped = {5, 6, 20, 21, 22}
    for seed in range(20):
        out = D._sample_block_idx(np.random.default_rng(seed), lo, hi, Bb, dropped)
        assert len(out) == Bb and len(set(out.tolist())) == Bb and not (set(out.tolist()) & dropped)
        assert out.min() >= lo and out.max() < hi and np.all(np.diff(out) > 0)
    try:
        D._sample_block_idx(np.random.default_rng(0), lo, hi, Bb, set(range(3, 45)))
        assert False, "must refuse when fewer than Bb directions remain"
    except AssertionError as e:
        assert "NPR" in str(e)


# ------------------------------------------------------------------ fp32 head
def test_fp32_head_hook_recomputes_logits_in_fp32():
    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
            self.lm_head = torch.nn.Linear(16, 64, bias=False, dtype=torch.bfloat16)
            self.lm_head.weight.requires_grad_(False)

        def forward(self, x):
            return self.lm_head(self.body(x))
    m = Tiny()
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    ref_bf16 = m(x)
    assert ref_bf16.dtype == torch.bfloat16
    h = D.install_fp32_head(m)
    out = m(x)
    hidden = m.body(x)
    assert out.dtype == torch.float32 and torch.allclose(out, hidden.float() @ m.lm_head.weight.float().T)
    assert not torch.equal(out, ref_bf16.float())                  # the bf16 rounding of the logits is gone
    out.sum().backward()
    assert m.body.weight.grad is not None and m.lm_head.weight.grad is None   # grad flows to the body, head stays frozen
    h.remove()
    assert m(x).dtype == torch.bfloat16
    m.lm_head.weight.requires_grad_(True)
    try:
        D.install_fp32_head(m); assert False, "a trainable head must be rejected"
    except AssertionError as e:
        assert "frozen" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("ALL OK")
