#!/usr/bin/env python3
"""Which exact tokens does the read average over? Decode one cell and mark them."""
import sys
sys.path.insert(0, "/workspace/inv/src")
import inv_core as C
from transformers import AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
G = C.Grid(tok, C.TEMPLATES_RECOVERED[:1], C.CARRIERS_RECOVERED[:1], 42, torch.zeros(1), "cpu")
S = G.cells[0][0]
slot = tok(" wanting to be comforted", add_special_tokens=False).input_ids
seq = S["pre"] + slot + S["post"]
n = S["ncar"]

print("=== the full sequence, one cell ===")
print(repr(tok.decode(seq)))
print()
print("=== token-by-token, with the READ span marked ===")
lo_slot, hi_slot = len(S["pre"]), len(S["pre"]) + len(slot)
read_lo = len(seq) - n
for i, t in enumerate(seq):
    tag = ""
    if lo_slot <= i < hi_slot:
        tag = "  <-- the candidate phrase (slot {x})"
    if i >= read_lo:
        tag = "  <=== READ (mean-pooled over these)"
    print("  %3d  %-22r%s" % (i, tok.decode([t]), tag))
print()
print("=== summary ===")
print("  pre  : %d tokens   ends %r" % (len(S["pre"]), tok.decode(S["pre"][-14:])))
print("  slot : %d tokens   = the candidate" % len(slot))
print("  post : %d tokens   = %r" % (len(S["post"]), tok.decode(S["post"])))
print("  read : last %d tokens = %r" % (n, tok.decode(seq[-n:])))
print()
gp = tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False,
                             add_generation_prompt=True, enable_thinking=False)
print("  generation prompt tail: %r" % gp[-40:])
print("  read span is AFTER the assistant header: %s"
      % ("<|im_start|>assistant" in tok.decode(S["post"])))
print("  carrier appears twice (quoted in the user turn, prefilled in the assistant turn): %s"
      % (tok.decode(seq).count("The chair stood near the window in the room.") == 2))
