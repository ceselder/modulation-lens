"""Blind pairwise judge: do the RL readouts beat the SFT warm start on activations the RL never saw?

Adapted from judge_targets.py -- same contract, same reason. Presentation order is randomised per
item so position cannot encode which pipeline produced a readout, and the judge never learns which
arm is which. Ties are reported, not silently split.

Each readout is 4 bullets describing one activation. The judge sees the text the model had just
read and picks the readout that better describes it.

  ANTHROPIC_API_KEY=... ANTHROPIC_WORKSPACE_ID=... \
  python rl/judge_holdout.py --a eval_holdout/av_sft_4b_final.json \
                             --b eval_holdout/ckpts_..._step_100.json --name-a sft --name-b rl100
"""
import argparse, json, os, random, re, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["ANTHROPIC_API_KEY"]
WS = os.environ["ANTHROPIC_WORKSPACE_ID"]
MODEL = "claude-sonnet-5"        # CLAUDE.md: Sonnet 5 for every judge call

PROMPT = """You are evaluating an interpretability tool that reads a language model's internal state.

At the moment we probed it, the model had just read this text:

<context>{ctx}</context>
<just_read>{label}</just_read>

Two candidate READOUTS were produced by two versions of the tool. Each is a list of short bullet
phrases meant to describe, together, WHAT THE MODEL'S INTERNAL STATE IS ABOUT at that moment.

Judge them on whether the bullets, taken together, identify the topic, situation or referent the
state is about. Penalise: generic filler that would fit any text, phrases that match only on
surface word-shape, bullets about something else entirely, and degenerate or non-linguistic output.
Do NOT reward fluency on its own -- a fluent readout about the wrong thing is worse than a rough
readout about the right thing.

<A>{a}</A>
<B>{b}</B>

Which readout better describes what the model's state is about? Reply with exactly one line:
VERDICT: A
VERDICT: B
VERDICT: TIE
Then one short sentence of justification."""


def ask(ctx, label, a, b, retries=6):
    body = json.dumps({
        "model": MODEL, "max_tokens": 200,
        "messages": [{"role": "user", "content": PROMPT.format(
            ctx=ctx[-700:], label=label, a=a, b=b)}],
    }).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body,
                headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                         "anthropic-workspace-id": WS, "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = json.loads(r.read())["content"][0]["text"]
            m = re.search(r"VERDICT:\s*([AB]|TIE)", txt)
            return (m.group(1) if m else None), txt
        except Exception as e:                       # low-prio key 429s are expected under load
            if k == retries - 1:
                return None, "ERROR %s" % str(e)[:120]
            time.sleep(min(2 ** k + random.random(), 30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--name-a", default="A"); p.add_argument("--name-b", default="B")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="")
    A = p.parse_args()

    ra = {r["i"]: r for r in json.load(open(A.a))}
    rb = {r["i"]: r for r in json.load(open(A.b))}
    shared = sorted(set(ra) & set(rb))[: A.n]
    if not shared:
        raise SystemExit("no overlapping probe rows between the two readout files")
    print("[judge] %d shared rows | %s vs %s" % (len(shared), A.name_a, A.name_b), flush=True)

    rng = random.Random(0)
    jobs = []
    for i in shared:
        flip = rng.random() < 0.5          # per-item order randomisation
        first, second = (rb[i], ra[i]) if flip else (ra[i], rb[i])
        jobs.append((i, flip, ra[i], rb[i], first["readout"], second["readout"]))

    def run(j):
        i, flip, x, y, s1, s2 = j
        v, txt = ask(x.get("ctx", ""), x.get("mark", ""), s1, s2)
        if v is None:
            win = None
        elif v == "TIE":
            win = "tie"
        else:                              # map the shown slot back to the arm that filled it
            shown_a_is_b = flip
            win = (A.name_b if (v == "A") == shown_a_is_b else A.name_a)
        return {"i": i, "winner": win, "verdict": v, "flip": flip,
                "label": x.get("mark", ""), "a": x["readout"], "b": y["readout"],
                "judge": txt.strip()[:400]}

    with ThreadPoolExecutor(max_workers=A.workers) as ex:
        rows = list(ex.map(run, jobs))

    n_a = sum(r["winner"] == A.name_a for r in rows)
    n_b = sum(r["winner"] == A.name_b for r in rows)
    n_t = sum(r["winner"] == "tie" for r in rows)
    n_e = sum(r["winner"] is None for r in rows)
    dec = n_a + n_b
    print("\n=== %s vs %s on %d held-out activations ===" % (A.name_a, A.name_b, len(rows)))
    print("  %-12s %d" % (A.name_a, n_a))
    print("  %-12s %d" % (A.name_b, n_b))
    print("  tie          %d\n  error        %d" % (n_t, n_e))
    if dec:
        frac = n_b / dec
        # binomial SE on the decided pairs; 0.5 means the RL arm did nothing measurable
        se = (frac * (1 - frac) / dec) ** 0.5
        print("  -> %s wins %.1f%% of %d decided pairs (SE %.1f%%)"
              % (A.name_b, 100 * frac, dec, 100 * se))
        print("     %s" % ("BEATS the baseline" if frac - 2 * se > 0.5 else
                           "LOSES to the baseline" if frac + 2 * se < 0.5 else
                           "indistinguishable from the baseline at 2 SE"))
    if A.out:
        json.dump(rows, open(A.out, "w"), indent=1)
        print("\nwrote", A.out)


if __name__ == "__main__":
    main()
