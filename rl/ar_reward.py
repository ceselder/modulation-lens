#!/usr/bin/env python3
"""The modulation-lens reward, as a drop-in for maemm's rl.py:score().

maemm's objective: inject a direction, the policy describes it, read the description back through
the CLEAN base model and take a position-max cosine against the direction.

Ours: the policy writes K bullets for a real L42 activation; each bullet is mapped to its
modulation vector by the FROZEN AR; the bullets are combined by exact NON-NEGATIVE least squares;
the reward is the cosine of that composition with the activation.

Three facts this encodes, each measured rather than assumed:

  * SPACE. Atoms are modulation reads (a phrase in a template, pooled over carrier positions);
    targets are natural activations at one position. Two different L42 distributions. Comparing
    them needs J *and* a fitted affine -- 4-atom FVE is 0.111 raw, 0.360 J-only, 0.349 affine-only,
    0.633 with both. Do NOT whiten: whitening costs 6.3x (0.633 -> 0.057).
  * The AR replaces 16 grid forwards per bullet with one. At 16 rollouts x 256 prompts x 4 bullets
    that is the difference between ~16k and ~262k 27B forwards per step.
  * The AR is a LoRA on the SAME base weights as the policy, so it is a second named adapter, not
    a second 27B.

The AR is accurate ON dictionary-like spans (held-out cos ~0.91) and degrades off-distribution
(~0.47 on a different construction, though still better than ridge's 0.32 there). RL explores
off-distribution by construction, so this reward must NOT be run alone -- keep the text term and
the KL anchor, and spot-check against the true grid reward.
"""
import os
import re
import numpy as np
import torch
import torch.nn.functional as F

_BULLET_RE = re.compile(r"^\s*(?:[*•\-–]|\d+[.)])\s+")


def split_bullets(text, k, max_tok, tok):
    """-> up to k bullet strings, each truncated to max_tok TOKENS.

    Truncating on tokens, not characters, because the AR reads tokens: a character cut can leave a
    half token that the AR never saw in training.
    """
    # NOTE for replay/eval callers: 11.6% of bank atoms contain an embedded newline
    # ('on Current Events\nOn August'), so joining atoms with '\n' and splitting on '\n' tears
    # them in two. A POLICY will not emit newlines mid-bullet, so this only bites when
    # reconstructing bullet text from the bank -- but that is exactly what calibration does.
    # Callers replaying bank atoms should pass them as a list, not as joined text.
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        s = _BULLET_RE.sub("", s).strip()
        if not s:
            continue
        ids = tok(s, add_special_tokens=False)["input_ids"][:max_tok]
        if not ids:
            continue
        s = tok.decode(ids, skip_special_tokens=True).strip()
        if s:
            out.append(s)
        if len(out) == k:
            break
    return out


def nnls_exact(B, t):
    """Exact non-negative least squares over a <=k support by enumerating every non-empty subset.

    B [k, d] rows, t [d]. Clamping an unconstrained lstsq is NOT exact -- it can return a point that
    is neither optimal nor feasible -- and k is small enough (<=4 -> 15 subsets) to enumerate.
    Returns (w [k], cos of the reconstruction with t).
    """
    import itertools
    k = B.shape[0]
    best_w, best_c = torch.zeros(k, device=B.device), -1.0
    G = B @ B.T
    c = B @ t
    for r in range(1, k + 1):
        for sup in itertools.combinations(range(k), r):
            idx = torch.tensor(sup, device=B.device)
            Gs = G[idx][:, idx] + 1e-6 * torch.eye(r, device=B.device)
            try:
                w = torch.linalg.solve(Gs, c[idx])
            except Exception:
                continue
            if bool((w < -1e-8).any()):
                continue
            w = w.clamp(min=0.0)
            rec = w @ B[idx]
            rn = rec.norm()
            if float(rn) <= 1e-8:
                continue
            cc = float((rec @ t) / rn)
            if cc > best_c:
                best_c = cc
                best_w = torch.zeros(k, device=B.device)
                best_w[idx] = w
    return best_w, max(best_c, 0.0)


def _distinct_fraction(input_ids, attention_mask):
    """Vectorised distinct-token fraction (special tokens included). Copied from maemm rl.py so the
    gate means the same thing under both objectives."""
    masked = input_ids.masked_fill(~attention_mask.bool(), -1)
    ordered = masked.sort(dim=1).values
    unique = torch.ones(len(input_ids), dtype=torch.long, device=input_ids.device)
    unique += (ordered[:, 1:] != ordered[:, :-1]).sum(1)
    unique -= (~attention_mask.bool()).any(1).long()
    return unique.float() / attention_mask.sum(1).clamp(min=1)


class ARReward:
    """Frozen text -> modulation-vector AR plus the J and affine transforms, loaded once."""

    def __init__(self, ar_dir, jlens_path, affine_path, device="cuda", read_layer=42,
                 max_tokens=12, adapter_name="ar", amu_path=""):
        self.dev = device
        self.read_layer = read_layer
        self.max_tokens = max_tokens
        self.adapter_name = adapter_name
        J = torch.load(jlens_path, map_location="cpu", weights_only=False)["J"][read_layer]
        self.J = J.to(device).float()
        self.M = torch.from_numpy(np.load(affine_path)).to(device).float()
        hd = torch.load(os.path.join(ar_dir, "head.pt"), map_location=device)
        D = self.J.shape[0]
        self.head = torch.nn.Linear(D, D, bias=True).to(device, torch.float32)
        self.head.load_state_dict(hd["head"])
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        # The activation-pool mean, subtracted in J-space. The bank holds RAW activations because
        # that is what gets INJECTED into the policy; the reward's comparison space is a different
        # transform of the same vector, so it is derived here rather than baked into the bank.
        # Two DIFFERENT means is the measured configuration: one shared mean puts a blank string at
        # 0.259 cosine, two means put it at 0.008.
        # WHITENER (optional, off by default). MEASURED 2026-09-04 on 20k bank activations: in the
        # mean-subtracted J space the best TARGET-BLIND constant direction scores cos 0.343, and a
        # 400-step run exploited exactly that -- from step 40 on, one of the four bullets was a
        # fixed phrase emitted verbatim for every activation ('* Spheres are unique in that every
        # point on their surface is'), soaking up the shared component while the other three did
        # the target-specific work. After whitening the same target-blind ceiling is 0.0064 (54x
        # lower), so the shortcut stops paying. Must be applied to BOTH sides of the cosine: the
        # affine M maps atoms into the UNWHITENED activation space, so whitening only the targets
        # would break the alignment.
        self.W = None
        self.amu = None
        if amu_path:
            z = np.load(amu_path)
            self.amu = torch.tensor(z["mu"] if hasattr(z, "files") else z,
                                    device=device).float()
        self.ar_dir = ar_dir
        self._hook_out = {}
        self._whiten_key = None
        self._handle = None

    def load_whitener(self, path, key="W_ridge0.1"):
        """Load an inverse-sqrt-covariance whitener for the J-space comparison.

        VERIFIED for natural_whitener_jspace.npz: both W_ridge0.01 and W_ridge0.1 are symmetric to
        <1e-10 and their eigenvalues equal 1/sqrt(eigval + lam) exactly, i.e. genuine C^-1/2 in
        J-space at layer 42 (n=60000, k90=490). eigvals.min()==0, so the un-ridged inverse does not
        exist -- use a ridge key, never a raw inverse.
        """
        z = np.load(path)
        if key not in z.files:
            raise SystemExit("whitener key %r not in %s (have %s)" % (key, path, z.files))
        W = torch.tensor(z[key], device=self.dev).float()
        if W.shape != (self.J.shape[0], self.J.shape[0]):
            raise SystemExit("whitener %s is %s, expected [%d,%d]"
                             % (key, tuple(W.shape), self.J.shape[0], self.J.shape[0]))
        self.W = W
        self._whiten_key = key
        return self

    def _maybe_whiten(self, x):
        return x if self.W is None else x @ self.W

    def build_own(self, base_model="Qwen/Qwen3.6-27B", dtype=None):
        """Load the AR on its OWN backbone, truncated to read_layer+1, with the adapter applied.

        Needed because the policy requires all layers to generate while the AR must be read on the
        truncation it was trained with -- sharing one base model is not an option (0.331 vs 0.759).
        Costs ~38 GB in bf16 for a 43-of-64-layer 27B, which is why this is a separate call and not
        the default: on a disaggregated setup put it on the trainer ranks, or give the reward its
        own GPU.

        Suspected mechanism for the truncation sensitivity: this model has hybrid GDN
        linear-attention layers carrying recurrent state, and training set
        config.num_hidden_layers = read_layer+1 after truncating. If per-layer state allocation
        keys off the layer count, an untruncated forward hands layer 42 a different state context.
        Unverified -- hence the enforcement rather than a fix.
        """
        import torch as _t
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        m, info = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype or _t.bfloat16, device_map={"": self.dev},
            output_loading_info=True)
        miss = [k for k in info.get("missing_keys", []) if "lora" not in k]
        if miss:
            raise SystemExit("AR base weights did not load: %s" % miss[:3])
        inner = m.model
        inner.layers = _t.nn.ModuleList(list(inner.layers[: self.read_layer + 1]))
        inner.config.num_hidden_layers = len(inner.layers)   # as ar_train_lora did
        del m.lm_head
        self.own = PeftModel.from_pretrained(inner, self.ar_dir).eval()
        for q in self.own.parameters():
            q.requires_grad_(False)
        n = sum(1 for k, _ in self.own.named_parameters() if "lora" in k)
        if n == 0:
            raise SystemExit("AR adapter loaded 0 LoRA tensors from %s" % self.ar_dir)
        stack = self.own.base_model.model
        stack = stack if hasattr(stack, "layers") else stack.model
        self._handle = stack.layers[self.read_layer].register_forward_hook(
            lambda mm, i, o: self._hook_out.__setitem__(
                "h", o[0] if isinstance(o, tuple) else o))
        print("[ar] own backbone: %d layers, %d lora tensors" % (len(stack.layers), n), flush=True)
        return self.own

    def attach(self, actor, require_truncated=True):
        """Load the AR as a SECOND named adapter on the policy's base weights and hook read_layer.

        REFUSES an untruncated backbone by default. MEASURED: the identical adapter, read through a
        64-layer Qwen3_5ForCausalLM instead of the 43-layer stack it was trained on, HALVES the
        reward -- 0.331 vs 0.759 on the same atoms and activations, with every internal path
        agreeing inside each config. A forward hook on layer 42 should not care what is above it and
        the mechanism is unexplained, so this is enforced rather than documented: silently getting
        half a reward is exactly the failure that survives a plausible-looking training curve.
        """
        inner0 = actor.base_model.model if hasattr(actor, "base_model") else actor.model
        stack = inner0 if hasattr(inner0, "layers") else inner0.model
        n_layers = len(stack.layers)
        if require_truncated and n_layers != self.read_layer + 1:
            raise SystemExit(
                "AR backbone has %d layers; it was trained on %d (truncated to read_layer+1). "
                "Reading it untruncated halves the reward (0.331 vs 0.759, measured). Truncate "
                "with `base.layers = base.layers[:%d]` BEFORE wrapping in PeftModel, or pass "
                "require_truncated=False if you have re-verified the calibration."
                % (n_layers, self.read_layer + 1, self.read_layer + 1))
        actor.load_adapter(self.ar_dir, adapter_name=self.adapter_name)
        n = sum(1 for k, _ in actor.named_parameters() if self.adapter_name in k)
        if n == 0:
            raise SystemExit("AR adapter loaded 0 tensors under name %r" % self.adapter_name)
        inner = actor.base_model.model if hasattr(actor, "base_model") else actor.model
        layers = inner.layers if hasattr(inner, "layers") else inner.model.layers
        self._handle = layers[self.read_layer].register_forward_hook(
            lambda m, i, o: self._hook_out.__setitem__(
                "h", o[0] if isinstance(o, tuple) else o))
        return n

    @torch.no_grad()
    def embed(self, phrases, actor, tok, batch=128):
        """-> [n, D] unit vectors in the TARGET comparison space (J then affine)."""
        own = getattr(self, "own", None)
        if own is not None:
            actor = own                      # its own truncated backbone; no adapter switching
            prev = None
        else:
            prev = getattr(actor, "active_adapter", None)
            actor.set_adapter(self.adapter_name)
        try:
            out = torch.empty((len(phrases), self.J.shape[0]), dtype=torch.float32, device=self.dev)
            for a in range(0, len(phrases), batch):
                b = tok(phrases[a:a + batch], add_special_tokens=False, padding=True,
                        truncation=True, max_length=self.max_tokens + 2,
                        return_tensors="pt").to(self.dev)
                actor(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
                h = self._hook_out["h"]
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                out[a:a + batch] = self.head(pooled.float())
        finally:
            if prev is not None:
                actor.set_adapter(prev)
        v = (out @ self.J.T) @ self.M.T
        return F.normalize(self._maybe_whiten(v), dim=-1)

    def target_space(self, acts):
        """RAW L42 activations [n, D] -> the reward's comparison space (J, minus AMU, unit).

        Kept separate from score() so the bank can hold raw activations (what the policy is
        INJECTED with) while the reward compares in J-space. Feeding already-transformed vectors
        here would apply J twice.
        """
        t = acts.to(self.dev).float() @ self.J.T
        if self.amu is not None:
            t = t - self.amu
        return F.normalize(self._maybe_whiten(t), dim=-1)

    @torch.no_grad()
    @torch.no_grad()   # inherited from score() when called there, but the percentile monitor calls
                       # this DIRECTLY -- without it, a 256-seq 27B forward builds a graph and OOMs.
    def fluency(self, texts, actor, tok, batch=64, max_len=128, need_logp=None):
        """-> (mean clean-base logp/token [n], distinct-token fraction [n]).

        WHY THIS EXISTS. The geometric reward alone is satisfied by illegible phrases -- measured:
        a 6-step run with --no-gates and kl=0 raised the reward 0.43 -> 0.65 while the rollouts
        degenerated into four mutually unrelated fragments plus CJK ('Omega Phoenix Quadratic
        Development of Quality of Mono Disaster Steel'). The policy learns to emit four
        high-variance directions whose non-negative combination spans the target, which is
        basis-fitting, not description. These are the gate inputs that penalise it.

        Scored on the ACTOR with its adapter DISABLED (the clean base), exactly as maemm's score()
        does -- not on the AR's own backbone, which is truncated and has no lm_head.
                CALIBRATION, MEASURED 2026-09-04. `logp` here is UNCONDITIONED -- raw text, no prompt,
        add_special_tokens=False -- so bullet fragments score far lower than conditioned prose
        would. A floor of -4.0 rejected 99.4% of perfectly legible warm-start rollouts
        (reward/gate_frac 0.0056), which under batch-normalized advantages is a near-constant
        offset that cancels while adding variance. Set any floor from percentiles of THIS
        distribution (see --flu-monitor-every), never from an absolute guess. The distinct-token
        floor needs no logits at all and measured min 0.706 on good text, so 0.6 is safe.

        need_logp=None follows self.need_logp (default True). When false the logits forward is
        SKIPPED entirely and logp comes back as zeros -- a sentinel that passes any floor -- so a
        distinct-only gate costs nothing.
        """
        n = len(texts)
        if need_logp is None:
            need_logp = getattr(self, "need_logp", True)
        logp = torch.full((n,), -20.0) if need_logp else torch.zeros(n)
        dis = torch.zeros(n)
        valid = [i for i, t in enumerate(texts) if (t or "").strip()]
        if not valid or actor is None:
            return logp, dis
        prev = tok.padding_side
        tok.padding_side = "right"
        try:
            for a in range(0, len(valid), batch):
                idxs = valid[a:a + batch]
                enc = tok([texts[i] for i in idxs], return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len,
                          add_special_tokens=False).to(self.dev)
                if enc["input_ids"].shape[1] < 2:
                    continue
                dis[idxs] = _distinct_fraction(enc["input_ids"],
                                               enc["attention_mask"]).cpu()
                if not need_logp:
                    continue          # distinct fraction needs token ids only, not a forward pass
                with actor.disable_adapter():
                    logits = actor(**enc).logits[:, :-1].float()
                tgt = enc["input_ids"][:, 1:]
                tlp = -F.cross_entropy(logits.flatten(0, 1), tgt.flatten(),
                                       reduction="none").view_as(tgt)
                nm = enc["attention_mask"].bool()[:, 1:]
                mlp = (tlp * nm).sum(1) / nm.sum(1).clamp(min=1)
                # rows with no next-token logprob keep -20 so they FAIL the floor
                logp[idxs] = torch.where(nm.any(1), mlp,
                                         torch.full_like(mlp, -20.0)).cpu()
        finally:
            tok.padding_side = prev
        return logp, dis

    @torch.no_grad()
    def score(self, texts, targets, actor, tok, k=4, max_tok=12, embed_batch=128,
              pre_split=None, targets_are_raw=True, with_fluency=False,
              contrast_negatives=0, contrast_weight=1.0, group_stride=1):
        """maemm score() contract: -> r [len(texts)] on CPU.

        targets [n, D]. With targets_are_raw=True (the default, and what the maemm bank supplies)
        they are RAW L42 activations and get transformed here; pass False only if the caller has
        already applied J and the mean.
        """
        n = len(texts)
        r = torch.zeros(n)
        # pre_split lets a caller supply bullets directly (list-of-lists), bypassing the parser.
        # Required when replaying bank atoms, which can contain newlines.
        bl = (list(pre_split) if pre_split is not None
              else [split_bullets(t, k, max_tok, tok) for t in texts])
        uniq = sorted({b for row in bl for b in row})
        if not uniq:
            return r
        emb = {p: i for i, p in enumerate(uniq)}
        V = self.embed(uniq, actor, tok, batch=embed_batch)
        tg = (self.target_space(targets) if targets_are_raw
              else F.normalize(targets.to(self.dev).float(), dim=-1))
        # CONTRASTIVE reward (contrast_negatives>0): r_i = fit(bullets_i, target_i)
        #   - contrast_weight * mean_j fit(bullets_i, target_{i+j+1 mod n})
        # The plain reconstruction reward is partly satisfiable WITHOUT reading the activation --
        # measured, a target-blind constant direction scores 0.343 in the mean-subtracted J space,
        # and a 400-step run duly spent one of its four bullets on a fixed phrase emitted for every
        # activation. Subtracting the fit against MISMATCHED targets credits only the part that
        # depends on which activation was injected, which is the quantity a lens is supposed to
        # carry. This is the permutation control promoted to being the objective.
        nb = []
        n_t = tg.shape[0]
        _fits, _negs_all = [], []      # instrumented: which TERM is off when the contrast collapses
        for i, row in enumerate(bl):
            if not row:
                continue
            B = V[torch.tensor([emb[p] for p in row], device=self.dev)]
            _, cc = nnls_exact(B, tg[i])
            if contrast_negatives > 0 and n_t > group_stride:
                # STRIDE BY THE GROUP SIZE. targets arrive as repeat_interleave(G), so `group_stride`
                # consecutive rows share ONE activation: offsetting by 1 would compare a readout
                # against its OWN target for (G-1)/G of the batch, making the contrast identically
                # zero for 94% of rollouts at G=16 and handing the zero-variance filter the entire
                # batch. The negative must come from a DIFFERENT group.
                negs = []
                for j in range(contrast_negatives):
                    k_off = (i + (j + 1) * group_stride) % n_t
                    if k_off // max(group_stride, 1) == i // max(group_stride, 1):
                        continue                      # same group -> same target, not a negative
                    _, cn = nnls_exact(B, tg[k_off])
                    negs.append(cn)
                if negs:
                    _negs_all.append(sum(negs) / len(negs))
                    cc = cc - contrast_weight * (sum(negs) / len(negs))
            r[i] = cc
            _fits.append(float(cc))
            nb.append(len(row))
        self.last_stats = {"mean_bullets": float(np.mean(nb)) if nb else 0.0,
                           "n_scored": len(_fits),
                           "mean_neg_fit": float(np.mean(_negs_all)) if _negs_all else float("nan"),
                           "mean_matched_fit": (float(np.mean(_fits) + contrast_weight * np.mean(_negs_all))
                                                if _negs_all else
                                                (float(np.mean(_fits)) if _fits else float("nan"))),
                           "frac_empty": float(sum(1 for x in bl if not x) / max(n, 1)),
                           "n_unique_bullets": len(uniq)}
        if with_fluency:
            lp, ds = self.fluency(texts, actor, tok)
            self.last_stats["mean_logp"] = float(lp.mean())
            self.last_stats["mean_distinct"] = float(ds.mean())
            return r, lp, ds
        return r
