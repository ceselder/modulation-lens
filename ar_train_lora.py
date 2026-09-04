"""Stage 1b: the AR proper -- atom text -> L42 modulation vector, LoRA on Qwen3.6-27B.

Design choices, each taken from a measured result rather than taste:
  * READ AT L42. The target IS an L42 quantity, and ar-readpos-head-ablation found read position is
    the only lever that moved reconstruction (+2.4 FVE) -- so match the target's layer. It also
    lets us drop layers 43..63 entirely, which is ~1/3 of the compute for free.
  * LINEAR HEAD on a MEAN-POOL. The same ablation clustered every head variant (bias, dedicated
    token, deeper MLP, fresh attention block) at 23.3-24.0 FVE: the head is not the bottleneck, so
    spending parameters there is wasted. Mean-pool matches the additive structure of the target
    (v(phrase) ~= sum of weighted token contributions).
  * COSINE loss, because the downstream use is a direction: the RL reward will be
    cos(AR(text), target_activation), so train the metric we will score with.
  * LR 1e-4, rank 64, alpha 16, rsLoRA -- project defaults; AR reconstruction specifically prefers
    1e-4 (futurelens-ar-lr-defaults: 1e-5 undertrains, inverted-U peak at 1e-4).
  * NO gradient checkpointing. On Qwen3.6-27B GDN it forces use_cache=False and crashes the fla
    chunk kernel; atoms are <=12 tokens so activation memory is negligible anyway.
"""
import os
import modal

app = modal.App("celeste-ar-lora")
VOL = modal.Volume.from_name("celeste-modlens-vol")
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("torch==2.8.0", "transformers==5.5.4", "numpy", "pyarrow", "safetensors",
                    "accelerate", "peft", "huggingface_hub", "wandb")
       .env({"HF_HOME": "/vol/.hf_home", "HF_HUB_OFFLINE": "1",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
MODEL = "Qwen/Qwen3.6-27B"
READ_LAYER = 42


@app.function(image=img, volumes={"/vol": VOL}, gpu="B200", cpu=8.0, memory=196608,
              timeout=86400, secrets=[modal.Secret.from_dict(
                  {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})])
def train(lr: float = 1e-4, rank: int = 64, alpha: int = 16, batch: int = 64,
          accum: int = 8, steps: int = 0, max_tokens: int = 12, eval_every: int = 200,
          variant: str = "meanpool_linear",
          out: str = "/vol/ar_l42_text2vec", wandb_project: str = "modlens-ar"):
    import json, time, numpy as np, torch, torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(MODEL)
    # Load through AutoModelForCausalLM (the checkpoint's key layout) and then take .model as the
    # backbone. AutoModel looked cleaner but builds a differently-prefixed skeleton that rejects
    # every checkpoint weight as UNEXPECTED and silently trains on random init.
    _full, info = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True)
    miss = [k for k in info.get("missing_keys", []) if "lora" not in k]
    if miss:
        raise SystemExit("REFUSING to train: %d weights did not load, e.g. %s"
                         % (len(miss), miss[:4]))
    print("[load] all weights loaded (0 missing, %d unexpected)"
          % len(info.get("unexpected_keys", [])), flush=True)
    model = _full.model                      # the backbone; skips the 248k-way lm_head entirely
    nlay = len(model.layers)
    model.layers = model.layers[:READ_LAYER + 1]
    model.config.num_hidden_layers = len(model.layers)
    del _full.lm_head
    print("[model] truncated %d -> %d layers (read L%d)" % (nlay, len(model.layers),
                                                            READ_LAYER), flush=True)
    # Read via a forward hook on layer READ_LAYER, exactly as the harvest and the playground do.
    # last_hidden_state would be norm(layer42_out) on a truncated stack, but the target vectors in
    # pg_dict came from the RAW layer output, so reading the normed version would train the AR on
    # a different quantity than it is asked to predict.
    HK = {"h": None}
    model.layers[READ_LAYER].register_forward_hook(
        lambda mm, i, o: HK.__setitem__("h", o[0] if isinstance(o, tuple) else o))

    lcfg = LoraConfig(r=rank, lora_alpha=alpha, use_rslora=True, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="FEATURE_EXTRACTION")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    D = int(model.config.hidden_size)
    # The baseline (meanpool_linear) decelerates to ~0.84 FVE with gains halving every 100 steps
    # while still seeing fresh atoms, so the limit is the map, not the data. These arms test the
    # two choices I made from a DIFFERENT task's ablation: the readout and the head.
    #   meanpool_linear : mean over the phrase's tokens -> Linear            (baseline)
    #   meanlast_linear : concat[mean, last token]      -> Linear            (readout capacity)
    #   meanpool_mlp    : mean                          -> Linear-GELU-Linear (head capacity)
    IN = D * 2 if variant == "meanlast_linear" else D
    if variant == "meanpool_mlp":
        head = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)).to(
            "cuda", torch.float32)
    else:
        head = nn.Linear(IN, D, bias=True).to("cuda", torch.float32)
        nn.init.zeros_(head.bias)
        if IN == D:
            nn.init.eye_(head.weight)     # identity prior; measured pre-train cos is only 0.05,
                                          # so this is harmless rather than helpful
    print("[variant] %s | head in %d -> out %d | params %s"
          % (variant, IN, D, f"{sum(q.numel() for q in head.parameters()):,}"), flush=True)

    labels = json.load(open("/vol/pg_dict/labels.json"))
    V = np.load("/vol/pg_dict/vectors.npy", mmap_mode="r")
    tr = np.load("/vol/ar_stage1/split_train_idx.npy")
    ev = np.load("/vol/ar_stage1/split_eval_idx.npy")
    rep = json.load(open("/vol/ar_stage1/ridge_report.json"))
    print("[data] train %s eval %s | ridge reference %.4f | cos bound %.4f"
          % (f"{len(tr):,}", f"{len(ev):,}", rep["ridge_cos"], rep["cos_bound"]), flush=True)

    import wandb
    wb = None
    if os.environ.get("WANDB_API_KEY"):
        wb = wandb.init(project=wandb_project, config=dict(
            lr=lr, rank=rank, alpha=alpha, batch=batch, max_tokens=max_tokens,
            read_layer=READ_LAYER, n_train=int(len(tr)), ridge_ref=rep["ridge_cos"],
            variant=variant))

    def encode(idx):
        txt = [labels[i] for i in idx]
        b = tok(txt, add_special_tokens=False, padding=True, truncation=True,
                max_length=max_tokens + 2, return_tensors="pt").to("cuda")
        return b

    def forward(b):
        model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
        h = HK["h"]
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)     # mean over real tokens only
        if variant == "meanlast_linear":
            idx = b["attention_mask"].sum(1).long() - 1          # last REAL token per row
            last = h[torch.arange(h.shape[0], device=h.device), idx]
            return head(torch.cat([pooled.float(), last.float()], dim=1))
        return head(pooled.float())

    # The eval split is stored SORTED by bank position, so ev[:n] is one CONTIGUOUS region of the
    # mining output -- effectively a single domain cluster, not a sample. Measured cost of that
    # mistake: the n=4096 prefix read 0.009 cos HIGHER than the full 20k set (SE at n=4096 is
    # ~0.002, so it was bias, not noise). Draw a fixed RANDOM subset instead, so periodic evals are
    # comparable to each other AND to the full-set number.
    _sub_rng = np.random.default_rng(7)
    _ev_shuf = ev[_sub_rng.permutation(len(ev))]

    @torch.no_grad()
    def evaluate(n=4096):
        model.eval(); tot, cnt = 0.0, 0
        pool = _ev_shuf if n < len(ev) else ev
        for a in range(0, min(n, len(pool)), 128):
            sub = np.sort(pool[a:a + 128])
            p = forward(encode(sub))
            y = torch.from_numpy(np.ascontiguousarray(V[sub])).to("cuda", torch.float32)
            c = ((p / p.norm(dim=1, keepdim=True).clamp(min=1e-8)) *
                 (y / y.norm(dim=1, keepdim=True).clamp(min=1e-8))).sum(1)
            tot += float(c.sum()); cnt += len(sub)
        model.train(); return tot / max(cnt, 1)

    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(1)
    order = rng.permutation(len(tr))
    # GRADIENT ACCUMULATION rather than one big micro-batch: with 43 layers live and no
    # gradient checkpointing (it crashes the fla chunk kernel on this model) a 512-example
    # micro-batch needed 178 GB. accum keeps the effective batch while paying 1/accum of the
    # activation memory.
    eff = batch * accum
    total = steps if steps > 0 else len(tr) // eff            # default: exactly one epoch
    print("[train] %d steps | micro-batch %d x accum %d = effective %d (1 epoch = %d steps)"
          % (total, batch, accum, eff, len(tr) // eff), flush=True)
    print("[eval] pre-training cos %.4f" % evaluate(2048), flush=True)

    t0 = time.time(); best = -1.0
    for st in range(total):
        lsum = 0.0
        for mi in range(accum):
            a0 = (st * eff + mi * batch) % len(tr)
            sel = np.sort(tr[order[a0:a0 + batch]])
            if len(sel) < 2: continue
            p = forward(encode(sel))
            y = torch.from_numpy(np.ascontiguousarray(V[sel])).to("cuda", torch.float32)
            pn = p / p.norm(dim=1, keepdim=True).clamp(min=1e-8)
            yn = y / y.norm(dim=1, keepdim=True).clamp(min=1e-8)
            loss = (1.0 - (pn * yn).sum(1)).mean() / accum
            loss.backward()
            lsum += float(loss) * accum
        loss = torch.tensor(lsum / max(accum, 1))
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if st % 25 == 0:
            print("[step %5d/%d] loss %.4f (train cos %.4f) | gnorm %.2f | %.1f ex/s"
                  % (st, total, float(loss), 1 - float(loss), float(gn),
                     (st + 1) * eff / max(time.time() - t0, 1e-9)), flush=True)
            if wb: wb.log({"loss": float(loss), "train_cos": 1 - float(loss),
                           "gnorm": float(gn), "step": st})
        if st > 0 and st % eval_every == 0:
            c = evaluate()
            flag = "  <-- beats ridge" if c > rep["ridge_cos"] else ""
            print("[eval  %5d] held-out cos %.4f (ridge %.4f, bound %.4f)%s"
                  % (st, c, rep["ridge_cos"], rep["cos_bound"], flag), flush=True)
            if wb: wb.log({"heldout_cos": c, "step": st})
            if c > best:
                best = c
                os.makedirs(out, exist_ok=True)
                model.save_pretrained(out)
                torch.save({"head": head.state_dict(), "read_layer": READ_LAYER,
                            "heldout_cos": c, "step": st, "max_tokens": max_tokens},
                           out + "/head.pt")
                json.dump({"variant": variant,
                           "heldout_cos": c, "step": st, "ridge_cos": rep["ridge_cos"],
                           "cos_bound": rep["cos_bound"], "lr": lr, "rank": rank,
                           "alpha": alpha, "batch": batch, "read_layer": READ_LAYER,
                           "max_tokens": max_tokens},
                          open(out + "/ar_meta.json", "w"), indent=1)
                VOL.commit()
    c = evaluate(len(ev))
    print("[final] full held-out cos %.4f | best checkpoint %.4f | ridge %.4f | bound %.4f"
          % (c, best, rep["ridge_cos"], rep["cos_bound"]), flush=True)
    if wb: wb.log({"final_cos": c}); wb.finish()
    print("AR_LORA_DONE %.4f" % max(best, c), flush=True)
    return max(best, c)
