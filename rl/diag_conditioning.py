"""Does the lens READ the injected activation, or does it just emit a good-on-average string?

Three numbers on held-out activations the RL never trained on:

  matched    reward(readout_i, target_i)          -- the normal reward
  permuted   reward(readout_i, target_{i+1})      -- same readouts, WRONG targets
  constant   reward(FIXED_STRING, target_i)       -- the string the 400-step run collapsed onto

`matched - permuted` is the only part of the reward that depends on having read the activation.
If it is ~0, the objective is satisfiable without conditioning and no amount of RL on it can
produce a lens. If `constant` is close to `matched`, a target-blind policy is already near-optimal,
which is exactly the collapse observed (reward 0.32 -> 0.85 with a single universal 4-bullet answer).

Run on the SFT warm start AND on a collapsed RL checkpoint to see how the gap moves.
"""
import json, os, modal   # numpy/torch imported inside run()

app = modal.App("modlens-diag-conditioning")
vol = modal.Volume.from_name("celeste-modlens-vol")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "transformers==5.15.0", "peft", "accelerate", "numpy",
                      "safetensors", "flash-linear-attention", "einops")
         .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir("src", "/root/src")
         .add_local_file("rl/ar_reward.py", "/root/ar_reward.py")
         .add_local_file("rl/diag_conditioning.py", "/root/diag.py"))

AR_DIR = "/vol/ar_frozen_rl_v1"
AFFINE = "/vol/data/affine_M_jspace.npy"
AMU = "/vol/data/natural_whitener_jspace.npz"
JLENS = ("/vol/.hf_home/hub/models--camilablank--workspace-lenses/snapshots/"
         "d740106d1e0f95456dc8718fba2895e9c8ffd6ef/qwen3.6-27b/j-lens/lens.pt")
PROMPT = "/vol/av_sft_4b/prompt.txt"
HOLDOUT = "/vol/rl_bank/vecs_holdout.f32"

# The string the 400-step run collapsed onto (step 200-260, cos ~0.85 on arbitrary targets).
COLLAPSED = ("* A sphere is unique in that every point on its surface is\n"
             "* Basil prefers consistent moisture, so it's\n"
             "* The auditors glanced at the reports,\n"
             "* VAT rates can be readily identified in the")


@app.function(image=image, volumes={"/vol": vol}, gpu="B200", timeout=7200)
def run(ckpt: str, n: int = 256, max_new: int = 96, whiten: str = "", whiten_key: str = "W_ridge0.1",
        temp: float = 0.0, samples: int = 1, probe: str = "", offset: int = 0,
        inject: str = "replace", unit_targets: bool = False):
    import sys, numpy as np, torch
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/src")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import inv_core as C
    import ar_reward as ARR

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16).to(dev).eval()
    actor = PeftModel.from_pretrained(m, ckpt).eval()
    n_lora = sum(1 for k, _ in actor.named_parameters() if "lora" in k)
    print("[load] %s | %d lora tensors | inject=%s" % (ckpt, n_lora, inject), flush=True)
    assert n_lora > 0, "adapter loaded 0 LoRA tensors"

    INJ, LEFT, RIGHT = C.marker_ids(tok)
    HOOK = {"ids": None, "vec": None}
    inner = m.model

    def _stash(mod, args, kwargs):
        HOOK["ids"] = kwargs.get("input_ids", args[0] if args else None)

    def _inject(mod, a, out):
        resid = out[0] if isinstance(out, tuple) else out
        ids, vec = HOOK["ids"], HOOK["vec"]
        if vec is None or ids is None or tuple(ids.shape) != tuple(resid.shape[:-1]):
            return out
        if not bool((ids == INJ).any()):
            return out
        # inject= selects the mode. THE TRAINING ROLLOUT USES KARVONEN (rl.py _steer_vec adds
        # unit(v)*hnorm*STEER_COEFF), while the SFT and every eval so far used REPLACE. inv_core's
        # own docstring: "A lens must be READ with the mode it was TRAINED with -- the two produce
        # different block-42 states, so mixing them yields confident garbage." This flag measures
        # whether that mismatch is what drove the trainer's matched_fit to 0.060.
        new = C.inject_at_marker(ids, resid, vec, INJ, LEFT, RIGHT, inject)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    inner.register_forward_pre_hook(_stash, with_kwargs=True)
    inner.layers[1].register_forward_hook(_inject)

    job = open(PROMPT).read()
    ptxt = tok.apply_chat_template([{"role": "user", "content": job}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    PIDS = torch.tensor(tok.encode(ptxt, add_special_tokens=False), device=dev)
    PLEN = PIDS.shape[0]
    at = (PIDS == INJ).nonzero().flatten()
    assert at.numel() == 1, "prompt needs exactly one marker, found %d" % at.numel()
    assert int(PIDS[int(at[0]) - 1]) == LEFT and int(PIDS[int(at[0]) + 1]) == RIGHT

    # probe= lets the SAME pipeline be pointed at the TRAINING bank instead of the holdout. The
    # trainer reports matched_fit 0.060 ~= neg_fit 0.059 (no information in the pairing) where this
    # diagnostic reports matched 0.395 vs permuted 0.181 on holdout rows. If bank rows also give
    # 0.395 here, the bank is fine and the trainer's readout<->target pairing is broken; if they
    # give ~0.06, the bank targets themselves are the problem.
    _src = probe or HOLDOUT
    _all = np.fromfile(_src, dtype="float32").reshape(-1, 5120)
    H = torch.from_numpy(_all[offset:offset + n]).float()
    if unit_targets:
        # Reproduce rl_disagg lines 1055/1160: the trainer L2-normalises bank rows for steering,
        # then hands those UNIT vectors to score(targets_are_raw=True), where target_space subtracts
        # a RAW-scale amu (raw ||h|| ~ 24). If -amu then dominates, every target collapses onto the
        # same vector and matched ~= negative -- which is what the trainer reports (0.060 / 0.059).
        H = torch.nn.functional.normalize(H, dim=-1)
        print("[unit_targets] bank rows L2-normalised, as the trainer does", flush=True)
    print("[probe] %s rows %d..%d of %d" % (_src, offset, offset + n, _all.shape[0]), flush=True)
    R = ARR.ARReward(AR_DIR, JLENS, AFFINE, device=dev, read_layer=42, max_tokens=12, amu_path=AMU)
    if whiten:
        # Does whitening keep the CONDITIONING (matched - permuted) while killing the target-blind
        # component? The affine was fit in the UNWHITENED space, so this is not free -- if matched
        # collapses too, whitening is the wrong fix and a 13h run would have found that out slowly.
        R.load_whitener(whiten, whiten_key)
        print("[whiten] %s[%s] applied to BOTH sides" % (whiten, whiten_key), flush=True)
    # build_own(): the AR must be read on its OWN 43-layer truncation. Do NOT attach() to the
    # policy backbone -- measured, that halves the reward (0.331 vs 0.759).
    R.build_own("Qwen/Qwen3.6-27B")

    texts = []
    with torch.no_grad():
        for s in range(0, H.shape[0], 8):
            sub = H[s:s + 8].to(dev)
            HOOK["vec"] = sub
            try:
                # temp>0 reproduces the TRAINING distribution. ScaleRL requires T=1.0, and the
                # whitened+contrastive reward came back at 0.002 on step 0 -- if sampled matched
                # ~= sampled permuted, that reward has no signal to optimise at all and the failure
                # is the temperature, not the reward space.
                gen = actor.generate(input_ids=PIDS.unsqueeze(0).expand(sub.shape[0], -1).contiguous(),
                                     attention_mask=torch.ones(sub.shape[0], PLEN, device=dev, dtype=torch.long),
                                     max_new_tokens=max_new, do_sample=temp > 0,
                                     temperature=temp if temp > 0 else None,
                                     top_p=1.0, top_k=0,
                                     pad_token_id=tok.eos_token_id)
            finally:
                HOOK["vec"] = None
            texts += [t.strip() for t in tok.batch_decode(gen[:, PLEN:], skip_special_tokens=True)]
    print("[gen] %d readouts | e.g. %r" % (len(texts), texts[0][:120]), flush=True)

    tg = H.to(dev)
    # ISOLATION TEST for the contrastive path. Here every row holds a DISTINCT activation, so
    # group_stride=1 makes the negative the next row = a genuinely different target. If this returns
    # ~= matched - permuted, the contrast arithmetic is correct and any collapse in training is
    # specific to the repeated-target group layout. If it returns ~0, the arithmetic is the bug.
    r_contrast = R.score(texts, tg, actor, tok, k=4, max_tok=12,
                         contrast_negatives=1, contrast_weight=1.0, group_stride=1)
    r_match = R.score(texts, tg, actor, tok, k=4, max_tok=12)
    r_perm = R.score(texts, torch.roll(tg, 1, dims=0), actor, tok, k=4, max_tok=12)
    r_const = R.score([COLLAPSED] * len(texts), tg, actor, tok, k=4, max_tok=12)
    # a second permutation to check the first was not a lucky alignment
    g = torch.Generator(device="cpu").manual_seed(0)
    perm2 = torch.randperm(tg.shape[0], generator=g)
    r_perm2 = R.score(texts, tg[perm2.to(dev)], actor, tok, k=4, max_tok=12)

    def st(x):
        return float(x.mean()), float(x.std() / max(len(x) ** 0.5, 1))

    out = {"ckpt": ckpt, "n": len(texts), "temp": temp, "inject": inject, "whiten": whiten or None, "whiten_key": whiten_key if whiten else None,
           "matched": st(r_match), "permuted_roll1": st(r_perm), "permuted_rand": st(r_perm2),
           "constant_collapsed_string": st(r_const),
           "delta_matched_minus_permuted": float(r_match.mean() - r_perm.mean()),
           "score_with_contrast_stride1": [float(r_contrast.mean()), float(r_contrast.std())],
           "sample_readouts": texts[:5]}
    print(json.dumps({k: v for k, v in out.items() if k != "sample_readouts"}, indent=1), flush=True)
    print("\n  contrast(score with contrast_negatives=1, stride 1) %.4f +- %.4f"
          % (float(r_contrast.mean()), float(r_contrast.std() / max(len(r_contrast) ** 0.5, 1))))
    print("  ... expected ~= matched - permuted if the arithmetic is right")
    print("\n  matched   %.4f +- %.4f" % out["matched"])
    print("  permuted  %.4f +- %.4f  (roll-1)" % out["permuted_roll1"])
    print("  permuted  %.4f +- %.4f  (random)" % out["permuted_rand"])
    print("  CONSTANT  %.4f +- %.4f  (the collapsed string)" % out["constant_collapsed_string"])
    print("\n  => reward attributable to READING the activation: %.4f"
          % out["delta_matched_minus_permuted"])
    os.makedirs("/vol/diag", exist_ok=True)
    tag = ckpt.rstrip("/").replace("/vol/", "").replace("/", "_") + ("_whitened" if whiten else "") + ("_T%g" % temp if temp > 0 else "_greedy") + ("_bank" if probe else "") + ("_" + inject) + ("_unit" if unit_targets else "")
    json.dump(out, open("/vol/diag/conditioning_%s.json" % tag, "w"), indent=1)
    vol.commit()
    return out


@app.local_entrypoint()
def main(ckpt: str = "/vol/av_sft_4b/final", n: int = 256, whiten: str = "", whiten_key: str = "W_ridge0.1",
         temp: float = 0.0, probe: str = "", offset: int = 0, inject: str = "replace",
         unit_targets: bool = False):
    run.remote(ckpt=ckpt, n=n, whiten=whiten, whiten_key=whiten_key, temp=temp, probe=probe,
               offset=offset, inject=inject, unit_targets=unit_targets)
