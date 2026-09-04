"""Which stage-2 targets should the AV be trained on: unaligned (FVE 0.36) or affine-aligned
(FVE 0.64)?

FVE says aligned, by 1.76x. Ten rows of eyeballing said aligned is often VAGUER ('in many
aspects'), and one match was a surface pun ('lush Platt Wood' -> 'plush sofa'). This project has
been burned by that gap before, so ask a judge which atom better describes what the model was
reading -- blind to which pipeline produced it, with the presentation order randomised per item so
position cannot encode the answer.
"""
import json, os, random, re, sys
import urllib.request

KEY = os.environ["ANTHROPIC_API_KEY"]
WS = os.environ["ANTHROPIC_WORKSPACE_ID"]
MODEL = "claude-sonnet-5"

PROMPT = """You are evaluating an interpretability tool that reads a language model's internal state.

At the moment we probed it, the model had just read this text:

<context>{ctx}</context>
<just_read>{label}</just_read>

Two candidate phrases were produced, each meant to describe WHAT THE MODEL'S INTERNAL STATE IS ABOUT
at that moment. A good phrase captures the topic, situation, or referent. A bad phrase is generic
filler, matches only on surface word-shape, or is about something else entirely.

<A>{a}</A>
<B>{b}</B>

Which phrase better describes what the model's state is about? Reply with exactly one line:
VERDICT: A
VERDICT: B
VERDICT: TIE
Then one short sentence of justification."""


def ask(ctx, label, a, b):
    body = json.dumps({
        "model": MODEL, "max_tokens": 200,
        "messages": [{"role": "user", "content": PROMPT.format(
            ctx=ctx[-700:], label=label, a=a, b=b)}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                 "anthropic-workspace-id": WS, "content-type": "application/json"})
    for attempt in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["content"][0]["text"]
        except Exception as e:
            if attempt == 4: return "ERROR %r" % e
            import time; time.sleep(2 ** attempt)


def main(n=120):
    old = {}
    with open("/tmp/nnomp_top1.jsonl") as f:
        for i, l in enumerate(f):
            if i >= 40000: break
            r = json.loads(l); old[r["i"]] = r
    new = {}
    with open("/tmp/nnomp_aff.jsonl") as f:
        for i, l in enumerate(f):
            if i >= 40000: break
            r = json.loads(l); new[r["i"]] = r
    # only items where the two pipelines DISAGREE -- identical targets carry no signal
    ids = [i for i in new if i in old and old[i]["bullets"][0] != new[i]["bullets"][0]]
    random.seed(11); random.shuffle(ids); ids = ids[:n]
    print("judging %d disagreements (of %d)" % (len(ids), len(
        [i for i in new if i in old and old[i]["bullets"][0] != new[i]["bullets"][0]])), flush=True)

    ctxs = {}
    with open("/tmp/prose_ctx.jsonl") as f:
        for l in f:
            r = json.loads(l); ctxs[r["i"]] = r["ctx"]
    tally = {"aligned": 0, "unaligned": 0, "tie": 0, "err": 0}
    for k, i in enumerate(ids):
        o, nw = old[i]["bullets"][0], new[i]["bullets"][0]
        flip = random.random() < 0.5                     # randomise presentation order
        a, b = (nw, o) if flip else (o, nw)
        out = ask(ctxs.get(i, ""), old[i]["label"], a, b)
        m = re.search(r"VERDICT:\s*([AB]|TIE)", out or "")
        if not m: tally["err"] += 1; continue
        v = m.group(1)
        if v == "TIE": tally["tie"] += 1
        else:
            picked_new = (v == "A") == flip
            tally["aligned" if picked_new else "unaligned"] += 1
        if (k + 1) % 20 == 0:
            print("  %d/%d -> %s" % (k + 1, len(ids), tally), flush=True)
    tot = tally["aligned"] + tally["unaligned"]
    print("\nFINAL %s" % tally, flush=True)
    if tot:
        print("aligned (FVE 0.64) preferred %.1f%% of decided pairs (n=%d)"
              % (100 * tally["aligned"] / tot, tot), flush=True)
    json.dump(tally, open("/tmp/judge_targets.json", "w"), indent=1)
    print("JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
