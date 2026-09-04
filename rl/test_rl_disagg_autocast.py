"""CPU unit test for rl_disagg's --autocast-bf16 policy-forward path (no GPU, no PEFT/HF needed).

A tiny module mimics PEFT's LoRA Linear exactly where it matters: a frozen bf16 base Linear, fp32 LoRA A/B master
weights (autocast_adapter_dtype=True behaviour), rsLoRA scaling as a Python float, lora_dropout = Identity, PEFT's
`cast_input_dtype_enabled` input-cast flag, a frozen bf16 `lm_head`, and rl_disagg's fp32 chunked log-softmax on top.

Checks: (1) flag OFF -> _policy_precision is a pure no-op: loss and grads bitwise identical to a plain forward;
(2) flag ON  -> loss within bf16 tolerance of the fp32-LoRA path, LoRA masters stay fp32 and receive fp32 grads,
    the input-cast flag is restored afterwards, and the LoRA matmul really ran in bf16 (the activation saved for
    backward is bf16);
(3) install_fp32_head composes: under autocast the head still returns fp32 logits equal to F.linear(x.float(), W.float()).

    python rl/test_rl_disagg_autocast.py      (or pytest)
"""
import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rl_disagg", os.path.join(_HERE, "rl_disagg.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

torch.manual_seed(0)
V, H, R, ALPHA = 64, 32, 8, 16.0


class LoraLinear(nn.Module):
    """PEFT lora.Linear forward, reduced to the arithmetic: base(x) + B(A(cast(x))) * scaling, cast back to base dtype."""
    cast_input_dtype_enabled = True

    def __init__(self, d_in, d_out):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=False).to(torch.bfloat16)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Linear(d_in, R, bias=False)            # fp32 masters (autocast_adapter_dtype=True)
        self.lora_B = nn.Linear(R, d_out, bias=False)
        nn.init.normal_(self.lora_B.weight, std=0.05)
        self.scaling = ALPHA / R ** 0.5                         # rsLoRA: alpha / sqrt(r), a Python float
        self.saved_dtype = None

    def forward(self, x):
        result = self.base(x)
        xx = x.to(self.lora_A.weight.dtype) if self.cast_input_dtype_enabled else x
        a_out = self.lora_A(xx)
        self.saved_dtype = a_out.dtype                          # dtype the LoRA matmul actually produced (== its saved activation)
        return (result + self.lora_B(a_out) * self.scaling).to(result.dtype)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, H).to(torch.bfloat16)
        self.emb.weight.requires_grad_(False)
        self.proj = LoraLinear(H, H)
        self.lm_head = nn.Linear(H, V, bias=False).to(torch.bfloat16)
        self.lm_head.weight.requires_grad_(False)

    def forward(self, ids):
        h = F.silu(self.proj(self.emb(ids)))
        return self.lm_head(h)


def _loss(model, ids, targets, autocast):
    with D._policy_precision(model, autocast):
        logits = model(ids)
    lp, ent = D._chunked_logp(logits.unsqueeze(0), targets.unsqueeze(0), vocab_chunk=3, need_entropy_grad=False)
    return -(lp.mean()), logits


def test_flag_off_is_bitwise_identical():
    m = Tiny(); ids = torch.randint(0, V, (7,)); tg = torch.randint(0, V, (7,))
    l0, lg0 = _loss(m, ids, tg, False); l0.backward()
    g0 = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    with torch.no_grad():
        logits_plain = m(ids)
    lp, _ = D._chunked_logp(logits_plain.unsqueeze(0), tg.unsqueeze(0), 3, False)
    assert torch.equal(lg0, logits_plain) and torch.equal(l0.detach(), -(lp.mean()))
    l1, _ = _loss(m, ids, tg, False); l1.backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.equal(p.grad, g0[n]), n
    assert m.proj.saved_dtype == torch.float32 and m.proj.cast_input_dtype_enabled is True


def test_autocast_matches_fp32_within_bf16_tolerance():
    m = Tiny(); ids = torch.randint(0, V, (11,)); tg = torch.randint(0, V, (11,))
    l32, lg32 = _loss(m, ids, tg, False); l32.backward()
    g32 = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    l16, lg16 = _loss(m, ids, tg, True); l16.backward()
    assert m.proj.saved_dtype == torch.bfloat16, "LoRA matmul did not run in bf16 under the autocast path"
    assert m.proj.cast_input_dtype_enabled is True, "input-cast flag not restored"
    assert lg16.dtype == lg32.dtype == torch.bfloat16
    rel = abs(float(l16) - float(l32)) / max(abs(float(l32)), 1e-6)
    assert rel < 2e-2, f"loss mismatch beyond bf16 tolerance: {float(l16)} vs {float(l32)} (rel {rel:.4f})"
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert p.dtype == torch.float32 and p.grad is not None and p.grad.dtype == torch.float32, n
            cos = F.cosine_similarity(p.grad.flatten(), g32[n].flatten(), dim=0).item()
            assert cos > 0.98, f"grad direction drifted under autocast for {n}: cos {cos:.4f}"


def test_fp32_head_composes_with_autocast():
    m = Tiny(); ids = torch.randint(0, V, (5,))
    handle = D.install_fp32_head(m)
    try:
        with D._policy_precision(m, True):
            logits = m(ids)
        assert logits.dtype == torch.float32
        with torch.no_grad():
            m.proj.cast_input_dtype_enabled = False
            with torch.autocast("cpu", dtype=torch.bfloat16):
                h = F.silu(m.proj(m.emb(ids)))
            ref = F.linear(h.float(), m.lm_head.weight.float())
        assert torch.allclose(logits, ref, atol=1e-5, rtol=1e-5)
    finally:
        handle.remove()
        m.proj.cast_input_dtype_enabled = True


def test_hook_runs_outside_autocast():
    seen = {}

    def hook(mod, inp, out):
        seen["autocast"] = torch.is_autocast_enabled("cpu")
        return out
    lin = nn.Linear(4, 4)
    h = lin.register_forward_hook(D._hook_outside_autocast(hook, True))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lin(torch.randn(2, 4))
    h.remove()
    assert seen["autocast"] is False
    assert D._hook_outside_autocast(hook, False) is hook


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("ALL OK")
