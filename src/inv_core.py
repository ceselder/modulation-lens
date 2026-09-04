#!/usr/bin/env python3
"""
Core pieces for the amortised inverter, reimplemented after the box holding nla/ was destroyed.

Three things live here because getting any of them subtly wrong produces plausible-looking garbage
rather than an error:

1. INJECTION. A single marker token in the prompt has its residual replaced by the target
   activation at an early layer. The original validated the marker's NEIGHBOURS and raised on a
   mismatch rather than no-op'ing, which is what caught a prompt where the marker sat on a bare line
   instead of inside <concept>...</concept>. That check is kept.

2. THE GRID / REWARD. A candidate phrase goes into the {x} slot of each template, the model is
   forced to write the carrier sentence, and layer 42 is read at those forced assistant-turn
   positions. Averaging over the grid is what cancels each template's idiosyncrasies -- a fixed
   4-of-16 subset mispriced phrases by +-50-80% and produced a formulaic-family artifact that
   vanished on the full grid. Carriers matter too: the same phrase against the same target spans
   sd 0.036 / range 0.119 across 8 carriers, so the carrier rotates per iteration and is held
   constant WITHIN a group so it cancels in the group-normalised advantage.

3. TWO-MEAN CENTRING. Candidates are centred by the grid's own mean (PMU, computed empirically for
   THIS grid), targets by the activation pool's mean (AMU). Both fixed constants, never the batch
   mean -- the batch is the policy's own rollouts, so subtracting its mean would cancel exactly the
   component being rewarded. With one shared mean a blank string scored 0.259 and search could not
   beat it; with two it scores 0.008.
"""
import glob, json, os
import numpy as np
import torch

D_MODEL = 5120
INJ_CHAR = "㈜"          # U+321C, id 158983. Single token; inside <concept>...</concept>
                            # its neighbours are 29 (">") and 510 ("</"), reproducing the
                            # published adapters' convention exactly, so those remain
                            # usable as a warm start. U+3234 looks nearly identical and is
                            # TWO tokens -- the single-token assert exists to catch that.


def load_jlens(layer, device):
    # Search every cache root this project runs under. The single hardcoded /workspace path
    # worked on the old box but does not exist on CA-MTL (cohort cache is
    # /workspace-vast/pretrained_ckpts) or on Modal (/vol/.hf_home), and this is loaded at
    # startup in EVERY mode -- including sft, which never touches J.
    roots = [os.environ.get("HF_HOME", ""), "/workspace/.hf_home",
             "/workspace-vast/pretrained_ckpts", "/vol/.hf_home",
             os.path.expanduser("~/.cache/huggingface")]
    p = []
    for r in roots:
        if not r:
            continue
        p = glob.glob(r + "/hub/models--camilablank--workspace-lenses/snapshots/*/"
                          "qwen3.6-27b/j-lens/lens.pt")
        if p:
            break
    if not p:
        raise SystemExit("j-lens not found under any of: %s" % [r for r in roots if r])
    return torch.load(p[0], map_location="cpu", weights_only=False)["J"][layer].to(device).float()


def load_whitener(path, ridge, device):
    z = np.load(path)
    if int(z.get("jtransformed", 0)) != 1:
        raise SystemExit("%s was not fitted in J-transformed space; the reward compares J-space "
                         "vectors and a raw-L42 whitener leaves a large shared constant in both "
                         "arguments, which is the loophole whitening exists to close" % path)
    return (torch.tensor(z["mu"], device=device),
            torch.tensor(z["W_ridge%s" % ridge], device=device))


def marker_ids(tok):
    """(marker, left, right) token ids. Karvonen injection requires the marker to sit between a
    specific pair, which is what forces <concept>MARKER</concept> rather than a bare line."""
    inj = tok(INJ_CHAR, add_special_tokens=False).input_ids
    assert len(inj) == 1, "marker glyph must be a single token, got %d" % len(inj)
    probe = tok("<concept>%s</concept>" % INJ_CHAR, add_special_tokens=False).input_ids
    k = probe.index(inj[0])
    return inj[0], probe[k - 1], probe[k + 1]


def inject_at_marker(ids, resid, vec, inj, left, right, mode="replace"):
    """Write vec into the residual at each validated marker site. ids [B,T], resid [B,T,d],
    vec [B,d] or [d]. Raises if no site has the right neighbours -- a silent no-op here is how a
    run trains for hours on an uninjected prompt.

    mode="replace"  : h'_p = v. Takes v's DIRECTION AND MAGNITUDE. Since we inject an L42 vector
                      at block 1 and residual norms grow with depth, this writes a magnitude that
                      belongs to a different depth.
    mode="karvonen" : h'_p = h_p + ||h_p|| * v/||v||. Takes only the DIRECTION and rescales to the
                      local norm, so the injected state stays on the block's own scale. This is
                      what the olens uses, and what the earlier modulation-lens work used.

    A lens must be READ with the mode it was TRAINED with -- the two produce different block-42
    states, so mixing them yields confident garbage."""
    B, T = ids.shape
    hit = (ids == inj)
    if T > 2:
        ok = torch.zeros_like(hit)
        ok[:, 1:-1] = hit[:, 1:-1] & (ids[:, :-2] == left) & (ids[:, 2:] == right)
    else:
        ok = torch.zeros_like(hit)
    n = int(ok.sum())
    if n == 0:
        raise RuntimeError("injection: 0 marker sites with correct neighbours (expected %d). The "
                           "marker must sit inside <concept>...</concept>." % B)
    v = vec if vec.dim() == 2 else vec.unsqueeze(0).expand(B, -1)
    out = resid.clone()
    bi, ti = ok.nonzero(as_tuple=True)
    if mode == "replace":
        out[bi, ti] = v[bi].to(out.dtype)
    elif mode == "karvonen":
        h = resid[bi, ti]                                   # [n, d], the local residual
        vv = v[bi].to(h.dtype)
        vv = vv / vv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        out[bi, ti] = (h + h.norm(dim=-1, keepdim=True) * vv).to(out.dtype)
    else:
        raise ValueError("unknown injection mode %r (replace|karvonen)" % mode)
    return out


class Grid:
    """templates x carriers, and the read that defines a phrase's vector."""

    def __init__(self, tok, templates, carriers, layer, J, device):
        self.tok, self.J, self.device, self.layer = tok, J, device, layer
        self.cells = []          # cells[carrier][template]
        for car in carriers:
            row = []
            for t in templates:
                body = t.replace("{x}", "XSLOT").replace("{y}", "ZSLOT")
                rend = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
                a, b = rend.replace("ZSLOT", car).split("XSLOT")
                cid = tok(car, add_special_tokens=False).input_ids
                row.append({"pre": tok(a, add_special_tokens=False).input_ids,
                            "post": tok(b, add_special_tokens=False).input_ids + cid,
                            "ncar": len(cid)})
            self.cells.append(row)
        self.n_car, self.n_tpl = len(carriers), len(templates)

    def sig(self):
        """Identity of this grid, hashed over the TOKENIZED cells rather than the template
        strings -- so a changed chat template or tokenizer also changes the signature. Used to
        key the PMU cache: PMU is the grid's own mean, and reusing one grid's mean under another
        grid silently rescores everything against the wrong centre."""
        import hashlib as _h, json as _j
        blob = _j.dumps([[[cell["pre"], cell["post"], cell["ncar"]] for cell in row]
                         for row in self.cells] + [self.layer], sort_keys=True)
        return _h.sha1(blob.encode()).hexdigest()

    @torch.no_grad()
    def read_all(self, model, strings, hook, max_tok=20, batch=48):
        """Average over EVERY (template, carrier) cell -- the clean modulation vector.

        One carrier per call gives a noisy estimate: the same phrase against the same target spans
        sd 0.036 / range 0.119 across carriers, which is the size of the differences the reward is
        trying to resolve. Rotating carriers across steps averages that out only in expectation and
        leaves each individual step noisy. Averaging all cells per rollout is the actual definition
        of the vector and is what the atom harvest did. Costs n_car x the forwards.
        """
        acc = None
        for ci in range(self.n_car):
            v = self.read(model, strings, hook, carrier=ci, max_tok=max_tok, batch=batch)
            acc = v if acc is None else {k: acc[k] + v[k] for k in acc}
        return {k: acc[k] / self.n_car for k in acc}

    @torch.no_grad()
    def read(self, model, strings, hook, carrier=0, max_tok=20, batch=48):
        """{string -> J-space vector}, averaged over every template of one carrier.

        Bucketed by token length: pre/post are fixed per cell and only the slot varies, so equal
        length strings form a rectangular batch with no padding inside the slot -- padding there
        would corrupt the read.
        """
        ids_of = {}
        for s in strings:
            t = self.tok(s, add_special_tokens=False).input_ids[:max_tok]
            ids_of[s] = t or self.tok(" the", add_special_tokens=False).input_ids
        acc = {s: torch.zeros(self.J.shape[0], device=self.device) for s in strings}
        buckets = {}
        for s, t in ids_of.items():
            buckets.setdefault(len(t), []).append(s)
        for S in self.cells[carrier % self.n_car]:
            pre = torch.tensor(S["pre"], device=self.device)
            post = torch.tensor(S["post"], device=self.device)
            for _, grp in buckets.items():
                for a in range(0, len(grp), batch):
                    ch = grp[a:a + batch]
                    mid = torch.tensor([ids_of[s] for s in ch], device=self.device)
                    B = mid.shape[0]
                    model(input_ids=torch.cat([pre.unsqueeze(0).expand(B, -1), mid,
                                               post.unsqueeze(0).expand(B, -1)], dim=1))
                    v = hook["h"].float()[:, -S["ncar"]:, :].mean(1) @ self.J.T
                    for k, s in enumerate(ch):
                        acc[s] += v[k]
        return {s: acc[s] / self.n_tpl for s in strings}

    @torch.no_grad()
    def prompt_mean(self, model, hook, n=64, seed=0, carrier=0):
        """PMU for THIS grid: the average read over many different slot fillers, so what remains
        after subtraction is what distinguishes one filler from another. The published thinkies
        ref_mean belongs to the old 16-template grid and would centre against the wrong thing."""
        import random
        rng = random.Random(seed)
        words = ("policy river garden engine harbour lantern meadow cipher tunnel orchard beacon "
                 "quarry saddle thistle vellum wharf pigment rafter cistern bramble").split()
        phrases = [" ".join(rng.choice(words) for _ in range(rng.randint(3, 9))) for _ in range(n)]
        # PMU must be measured on the same cells the candidates are: the whole grid.
        v = self.read_all(model, phrases, hook)
        return torch.stack([v[p] for p in phrases]).mean(0)


def chat_wrap_ids(tok):
    """(prefix, suffix) token ids for a user turn, split on a sentinel so a passage's own tokens are
    untouched and position i of the passage stays position i. Reads must be chat-native: raw-vs-chat
    differs by whitened cosine 0.75 averaged over positions and as low as 0.06 at the worst."""
    r = tok.apply_chat_template([{"role": "user", "content": "XSLOT"}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    a, b = r.split("XSLOT")
    return (tok(a, add_special_tokens=False).input_ids,
            tok(b, add_special_tokens=False).input_ids)


# 6 of the original 16 templates, recovered verbatim from the session transcript before the box
# holding templates_final.json was destroyed. The other 10 are lost, so this is a NEW grid: numbers
# from it are not comparable to anything measured on the old one, and the published thinkies-v3
# atom vectors cannot be reproduced in this geometry.
# --- GRID v2, selected 2026-08-25 -------------------------------------------------------------
# The v1 grid was 6 of the original 16 templates (10 lost with the old box) and it was WEAK: five of
# the six were near-paraphrases of "«verb» X while writing Y", bare-content readouts scored 0.18-0.25,
# and a boilerplate wrapper the RL policy invented ("not pizza related wording but related X") bought
# +0.079 -- it was compensating for the template, not gaming the metric.
#
# What screening 130+ candidates established:
#   * |mod| (modulation magnitude) correlates +0.71 with discrimination and -0.87 with wrapper bonus
#   * RELATIONAL framing beats IDENTIFYING framing by +0.026..+0.042 at every foil level:
#     "focus on something related to: X"  >>  "focus on the following idea: X"
#   * y-FIRST ordering wins: top 8 of 107 screened all state the carrier first
#   * demanding VERBATIM reproduction keeps the carrier near-certain (NLL ~0.00), which is wanted:
#     a carrier that is a real choice makes the model spend capacity choosing instead of holding X
#   * bind X to the act of writing (+0.046 over isolating it in its own sentence)
#   * SHORT: elaboration hurts. 7 framing tokens get 98% of what 15 do.
# Result: bare 0.357 -> 0.396, wrapper bonus +0.034 -> +0.017 at equal length.
TEMPLATES_RECOVERED = [
 'Repeat this line verbatim: "{y}"\n\nAs you write it, focus on something related to: "{x}". '
 'Output only the sentence.',
 'Repeat this line verbatim: "{y}"\n\nAs you write it, focus on something connected to: "{x}". '
 'Output only the sentence.',
 'Write out this line exactly: "{y}"\n\nWhile writing it, focus on something related to: "{x}". '
 'Output only the line.',
 'Copy this sentence word for word: "{y}"\n\nAs you copy it, really focus on something related '
 'to: "{x}". Output only the sentence.',
 'Reproduce this line verbatim: "{y}"\n\nAs you write it, focus on something in the area of: '
 '"{x}". Output only the line.',
 # one foil variant: the only family member measured with a NEGATIVE wrapper bonus (-0.006), i.e.
 # padding actively costs score. The noun is arbitrary -- pizza/cheese/bicycle differ by <0.004.
 'Repeat this line verbatim: "{y}"\n\nAs you write it, focus on something that is not cheese '
 'related wording but is related to: "{x}". Output only the sentence.',
]
# v1, kept for reproducing anything measured before 2026-08-25
TEMPLATES_V1 = [
 'Focus on the following idea: "{x}" while writing the following phrase: "{y}"\n\n'
 "Write only the last phrase. Don't respond with anything else",
 'concentrate on "{x}" while you write "{y}". Respond only with "{y}", nothing else.',
 'Think deeply about "{x}" while replying with nothing but "{y}".',
 'Deeply reflect on "{x}" while responding with just the following sentence: "{y}".',
 'Keep thinking of "{x}" as you compose your reply, limiting it strictly to "{y}".',
 'Hold the concept of "{x}" in your thoughts, then respond with just "{y}".',
]
# The same six templates with ONE change: the focus instruction acknowledges that the phrase may
# sound arbitrary. Motivation: readouts hedge when the policy cannot pin the content down -- 3.0% of
# emitted lines contain "random" (concentrated on a few activations) and 20.4% retreat to describing
# the medium ("a snippet from a conversation transcript") instead of the content. If the READER is
# told the phrase may sound random, the policy no longer has to spend its budget apologising for it.
#
# Kept as a parallel list rather than a mutation of TEMPLATES_RECOVERED so the A/B is one variable:
# identical carriers, identical wording, identical order, only the acknowledgement differs. Note this
# changes the grid signature, hence the PMU cache key, which is correct -- PMU is the grid's own mean
# and reusing the old one would score everything against the wrong centre.
TEMPLATES_RANDOM_ACK = [
    t.replace(' to: "{x}"', ' to this, which I know may sound random: "{x}"')
     .replace(' of: "{x}"', ' of this, which I know may sound random: "{x}"')
    for t in TEMPLATES_RECOVERED
]

CARRIERS_RECOVERED = [
 'The chair stood near the window in the room.',
 'A small clock ticked on the shelf quietly.',
 'The book rested on the wooden table.',
 'Rain fell softly against the glass window.',
 'The car was parked outside the house.',
 'A bird sat on the fence for hours.',
]
