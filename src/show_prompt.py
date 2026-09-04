#!/usr/bin/env python3
"""Print the rendered four-bullet JOB prompt for a given config, without loading the 27B.

The prompt is assembled by a chain of string replacements in inv_train.py, and a prompt/checkpoint
mismatch has already cost a round of malformed rollouts once. Rather than re-implement the chain
here (which is how the two drift apart), lift the exact source block out of inv_train.py and exec it
against a stub config.
"""
import argparse, re, sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/inv/src")
import inv_core as C
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--bullets", type=int, default=4)
ap.add_argument("--bullet-max-tok", type=int, default=10)
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--inject", default="karvonen")
a = ap.parse_args()
A = SimpleNamespace(bullets=a.bullets, bullet_max_tok=a.bullet_max_tok, max_new=a.max_new,
                    inject=a.inject)

src = open("/workspace/inv/src/inv_train.py").read()
start = src.index('JOB = ("You are shown an internal activation vector')
end = src.index("PROMPT_TXT = tok.apply_chat_template")
block = src[start:end]
ns = {"A": A, "C": C}
exec(block, ns)
JOB = ns["JOB"]
print("=" * 100)
print(JOB)
print("=" * 100)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
txt = tok.apply_chat_template([{"role": "user", "content": JOB}], tokenize=False,
                              add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(txt, add_special_tokens=False)
INJ, LEFT, RIGHT = C.marker_ids(tok)
at = [i for i, t in enumerate(ids) if t == INJ]
print("chat-templated: %d tokens | marker id %d at %s | neighbours %s/%s"
      % (len(ids), INJ, at, ids[at[0] - 1] if at else None, ids[at[0] + 1] if at else None))
assert len(at) == 1, "marker count %d" % len(at)
assert ids[at[0] - 1] == LEFT and ids[at[0] + 1] == RIGHT, "marker neighbours wrong"
print("chars %d | per-line cap stated: %s" % (len(JOB), "%d tokens PER LINE" % A.bullet_max_tok
                                              in JOB))
print("SHOW_PROMPT_OK")
