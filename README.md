# modulation lens

A lens that reads one activation of a language model and describes it in **four short natural-language
bullets**, instead of the single tokens a logit- or Jacobian-lens can emit.

Built on `Qwen/Qwen3.6-27B`, reading layer 42.

```
"The defendant argued that the evidence had been obtained without a warrant,"

  at token 5  ("...argued that the evidence")
    * evidence to prove | * of the evidence presented at trial | * of the evidence in a criminal case
  at the final token ("...without a warrant,")
    * a violation of the Fourth Amendment | * is not valid
    | * to argue that the evidence obtained by the police was
```

"Fourth Amendment" is nowhere in the input. It is read out of the state.

## Does it actually read the activation?

That is the only question worth asking of a lens, and it is answered with a **permutation control**:
score each readout against its own activation (`matched`) and against a *different* one (`permuted`).
The gap is the part that required reading the input.

| checkpoint | matched | permuted | delta |
|---|---|---|---|
| SFT warm start | 0.7120 | 0.2352 | 0.4768 |
| RL step 25 | -- | 0.1563 | **0.5798** |
| RL step 50 | -- | 0.1499 | **0.5927** |

256 reserved holdout activations the RL never trained on, greedy decoding. `permuted` *falling* is the
meaningful half: readouts became harder to match to the wrong activation, i.e. more specific rather
than more plausible-sounding.

On the mechanical half of [workspace-bench](https://huggingface.co/datasets/camilablank/workspace-bench)
(10 banks x 100 items, deterministic scoring) the lens scores **0.383 mean vs 0.196 for a j-lens
baseline**, and on the four *multi-token* banks the j-lens scores **0.000** -- a single token cannot be
a multi-token target, which is the gap this lens exists to fill. It does NOT lead that leaderboard:
several earlier arms score 0.50-0.60, so treat 0.383 as a working system, not a best result.

## Layout

```
src/            lens training and inference (inv_train.py, av_readout.py, inv_core.py)
rl/             RL: vendored ScaleRL trainer + the modulation-lens objective
  ar_reward.py       text -> vector AR reward, J-space, mean-centred, optional whitening
  rl_disagg.py       disaggregated GRPO/CISPO (vLLM rollout GPUs + HF trainer GPUs)
  diag_*.py          the diagnostics that caught the bugs below -- read these first
  playground_*.py    interactive web playground
  blogpost_modlens.py / modal_wsbench_gen.py   evaluation harnesses
mxf/            injection hooks and shared config
scripts/        plotting
```

## Four things that silently produce a wrong number

Each of these cost real GPU time here, and each returns a plausible result rather than an error.

1. **Never L2-normalise the target activation.** The reward subtracts a raw-scale mean in J-space, so
   unit vectors collapse every target onto `-amu`: a FIXED string then outscores correct readouts 3:1
   and RL correctly learns to ignore the input. `target_space()` now refuses unit-scale inputs.
2. **Read the lens with the injection mode it was trained with.** replace (`h'_p = v`) vs karvonen
   (`h_p + ||h_p||*unit(v)`) differ by 34% of the conditioning delta.
3. **Re-apply the chat template** to the bundled `prompt.txt`. Encoding it raw gives 174 tokens
   instead of 186 and greedy decoding emits EOS immediately -- every readout comes back empty.
4. **A green plumbing check is not a correct feed.** The injection verifier used a *random* vector, so
   it reported `cos=1.0000` while the reward was being fed collapsed targets for two entire runs.

## Reward-vs-readability

The training reward is a composition cosine over the four bullets, and it cannot see legibility. With
no KL anchor the policy sharpens (entropy 3.66 -> 1.16) and past ~step 30 bullets degrade into
multilingual token salad *while the reward keeps rising*. Non-ASCII fraction of rollouts:

```
step 0 (SFT) 0.0258 | step 10 0.0013 | step 20 0.0000 | step 30 0.0012 | step 40 0.0206
```

So `step_25` is the readable checkpoint and `step_50` the higher-delta one. Select on held-out
readouts, never on reward.

## Checkpoints

- `ceselder/modulation-lens-4bullet-rl-step25` -- best readable
- `ceselder/modulation-lens-4bullet-rl-step50` -- best delta

## Credits

The J-lens and the workspace framing are Anthropic's
([global workspace](https://www.anthropic.com/research/global-workspace)). The RL trainer is vendored
from `ceselder/maemm` with only the reward swapped; injection runs through
[vllm-lens](https://pypi.org/project/vllm-lens/). Item banks from
`camilablank/workspace-bench` (MIT).

MIT licensed.
