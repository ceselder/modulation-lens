"""Shared prompt: a single ` ?` marker whose residual gets the injected direction at INJECT_LAYER."""
from mxf.config import READ_LAYER

MARKER = " ?"
# Inoculation-style framing: the task is scoped as a research tool for characterizing a probe
# direction, NOT an unconditional "emit weird activating text" behavior. Explicit purpose keeps the
# finetuned behavior conditioned on this context so it does not leak into the model's defaults.
_INSTR = (
    "You are an interpretability research tool. Researchers have injected a single linear probe "
    f"direction from this model's own layer-{READ_LAYER} residual stream. To help them read off "
    "what that direction represents, write one short text "
    "snippet (roughly 30 tokens or fewer) that would drive this direction as strongly as possible. "
    "Output only the snippet itself, with no explanation, preamble, or quotation marks. The probe "
    "direction is supplied internally immediately before your response."
)


def _chat_ids(tok, content, add_gen):
    out = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                  add_generation_prompt=add_gen, enable_thinking=False)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    while isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def marker_positions(tok, ids):
    mid = tok.encode(MARKER, add_special_tokens=False)
    assert len(mid) == 1, f"marker not single-token: {mid}"
    pos = [i for i, t in enumerate(ids) if t == mid[0]]
    assert len(pos) == 1, f"expected exactly one marker, got {len(pos)}"
    return pos


def build_prompt_ids(tok):
    # Append the masked marker *after* the chat template's assistant-generation prefix.  Putting it
    # at the end of the user message still leaves nine Qwen chat-control tokens before generation;
    # the previous mid-prompt marker left 61 instruction tokens and empirically erased conditioning.
    ids = _chat_ids(tok, _INSTR, add_gen=True) + tok.encode(MARKER, add_special_tokens=False)
    return ids, [len(ids) - 1]


def build_sft_ids(tok, target_text):
    """Full sequence prompt+target, labels masked on the prompt. Returns (ids, labels, positions)."""
    prompt_ids, positions = build_prompt_ids(tok)
    tgt = tok.encode(target_text, add_special_tokens=False) + [tok.eos_token_id]
    ids = prompt_ids + tgt
    labels = [-100] * len(prompt_ids) + tgt
    return ids, labels, positions
