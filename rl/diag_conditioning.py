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
def run(ckpt: str, n: int = 256, max_new: int = 96):
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
    print("[load] %s | %d lora tensors" % (ckpt, n_lora), flush=True)
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
        new = C.inject_at_marker(ids, resid, vec, INJ, LEFT, RIGHT, "replace")
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

    H = torch.from_numpy(np.fromfile(HOLDOUT, dtype="float32").reshape(-1, 5120)[:n]).float()
    R = ARR.ARReward(AR_DIR, JLENS, AFFINE, device=dev, read_layer=42, max_tokens=12, amu_path=AMU)
    # build_own(): the AR must be read on its OWN 43-layer truncation. Do NOT attach() to the
    # policy backbone -- measured, that halves the reward (0.331 vs 0.759).
    R.build_own("Qwen/Qwen3.6-27B")

    texts = []
    with torch.no_grad():
        for s in range(0, H.shape[0], 8):
            sub = H[s:s + 8].to(dev)
            HOOK["vec"] = sub
            try:
                gen = actor.generate(input_ids=PIDS.unsqueeze(0).expand(sub.shape[0], -1).contiguous(),
                                     attention_mask=torch.ones(sub.shape[0], PLEN, device=dev, dtype=torch.long),
                                     max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            finally:
                HOOK["vec"] = None
            texts += [t.strip() for t in tok.batch_decode(gen[:, PLEN:], skip_special_tokens=True)]
    print("[gen] %d readouts | e.g. %r" % (len(texts), texts[0][:120]), flush=True)

    tg = H.to(dev)
    r_match = R.score(texts, tg, actor, tok, k=4, max_tok=12)
    r_perm = R.score(texts, torch.roll(tg, 1, dims=0), actor, tok, k=4, max_tok=12)
    r_const = R.score([COLLAPSED] * len(texts), tg, actor, tok, k=4, max_tok=12)
    # a second permutation to check the first was not a lucky alignment
    g = torch.Generator(device="cpu").manual_seed(0)
    perm2 = torch.randperm(tg.shape[0], generator=g)
    r_perm2 = R.score(texts, tg[perm2.to(dev)], actor, tok, k=4, max_tok=12)

    def st(x):
        return float(x.mean()), float(x.std() / max(len(x) ** 0.5, 1))

    out = {"ckpt": ckpt, "n": len(texts),
           "matched": st(r_match), "permuted_roll1": st(r_perm), "permuted_rand": st(r_perm2),
           "constant_collapsed_string": st(r_const),
           "delta_matched_minus_permuted": float(r_match.mean() - r_perm.mean()),
           "sample_readouts": texts[:5]}
    print(json.dumps({k: v for k, v in out.items() if k != "sample_readouts"}, indent=1), flush=True)
    print("\n  matched   %.4f +- %.4f" % out["matched"])
    print("  permuted  %.4f +- %.4f  (roll-1)" % out["permuted_roll1"])
    print("  permuted  %.4f +- %.4f  (random)" % out["permuted_rand"])
    print("  CONSTANT  %.4f +- %.4f  (the collapsed string)" % out["constant_collapsed_string"])
    print("\n  => reward attributable to READING the activation: %.4f"
          % out["delta_matched_minus_permuted"])
    os.makedirs("/vol/diag", exist_ok=True)
    tag = ckpt.rstrip("/").replace("/vol/", "").replace("/", "_")
    json.dump(out, open("/vol/diag/conditioning_%s.json" % tag, "w"), indent=1)
    vol.commit()
    return out


@app.local_entrypoint()
def main(ckpt: str = "/vol/av_sft_4b/final", n: int = 256):
    run.remote(ckpt=ckpt, n=n)
