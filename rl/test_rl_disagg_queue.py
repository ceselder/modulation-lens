"""CPU unit tests for train/rl_disagg.py's filesystem plumbing (no GPU, no vLLM, no HF model):
rollout-block queue (FIFO / drop-stale), adapter `latest` pointer, and the inline-eval request -> shard
protocol (request layout, per-rank chunk planning, shard readiness, merge).

    python -m pytest train/test_rl_disagg_queue.py -q      (or: python train/test_rl_disagg_queue.py)
"""
import importlib.util
import os
import tempfile
import time
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rl_disagg", os.path.join(_HERE, "rl_disagg.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def _touch(path):
    open(path, "w").close()
    time.sleep(0.002)


def test_parse_args_block_geometry():
    a = D.parse_args(["--role", "launch", "--n-rollout", "1", "--n-trainer", "3", "--groups-per-step", "128", "--group-size", "8"])
    assert (a.rollout_block_groups, a.blocks_per_step, a.max_num_seqs, a.max_queue_blocks) == (128, 1, 1024, 1)
    a = D.parse_args(["--role", "launch", "--n-rollout", "3", "--n-trainer", "1", "--groups-per-step", "128", "--group-size", "8",
                      "--rollout-block-groups", "64"])
    assert (a.rollout_block_groups, a.blocks_per_step, a.max_num_seqs, a.max_queue_blocks) == (64, 2, 512, 2)


def test_pick_blocks_fifo_and_drop_stale():
    w = tempfile.mkdtemp()
    os.makedirs(f"{w}/queue")
    for k in (3, 1, 2, 1):
        _touch(f"{w}/queue/blk_{k:07d}_{time.time_ns()}_0.pt")
    take, stale = D._pick_blocks(w, 2, False)
    assert [os.path.basename(x)[:11] for x in take] == ["blk_0000001", "blk_0000001"] and stale == []
    take, stale = D._pick_blocks(w, 2, True)
    assert [os.path.basename(x)[:11] for x in take] == ["blk_0000002", "blk_0000003"] and len(stale) == 2
    assert D._pick_blocks(w, 9, False) == (None, [])
    os.makedirs(f"{w}/lora")
    D._atomic_write_text(f"{w}/lora/latest", "7")
    assert D._read_latest(w) == 7
    assert D._read_latest(tempfile.mkdtemp()) is None


def _fake_request(n_fam_rows=10, n_sae=6, n_extra=5, d=8):
    sets = [{"name": "realact", "kind": "cos", "n_rows": n_fam_rows, "dirs": torch.randn(n_fam_rows, d), "n": 4, "temp": 1.0,
             "min_new": 16, "max_new": 64, "seed_base": 1234 * 1000},
            {"name": "sae", "kind": "sae", "n_rows": n_sae, "dirs": torch.randn(n_sae, d), "n": 4, "temp": 1.0,
             "min_new": 16, "max_new": 64, "seed_base": 1234 * 1000, "feats": list(range(n_sae))},
            {"name": "extra", "kind": "extra", "n_rows": n_extra, "dirs": torch.randn(n_extra, d), "n": 3, "temp": 1.0,
             "min_new": 16, "max_new": 64, "seed_base": 4321 * 1000, "feats": [100 + i for i in range(n_extra)]}]
    return {"ckpt_step": 10, "adapter_step": 11, "t": time.time(), "sets": sets}


def test_eval_plan_covers_every_row_once_across_ranks():
    req = _fake_request()
    X = 3
    seen = {st["name"]: [] for st in req["sets"]}
    for r in range(X):
        for ch in D._eval_plan(req, r, X, chunk_seqs=8):
            assert len(ch["rows"]) * ch["n"] <= 8 or len(ch["rows"]) == 1
            assert all(i % X == r for i in ch["rows"])
            assert ch["seeds"] == [(st["seed_base"] + i) % 2147483647 for st in req["sets"] if st["name"] == ch["set"] for i in ch["rows"]]
            seen[ch["set"]] += ch["rows"]
    for st in req["sets"]:
        assert sorted(seen[st["name"]]) == list(range(st["n_rows"])), st["name"]


def test_eval_request_shard_roundtrip():
    w = tempfile.mkdtemp()
    os.makedirs(f"{w}/eval_req"); os.makedirs(f"{w}/eval_gen")
    assert D._eval_requests(w) == []
    req = _fake_request()
    torch.save(req, D._eval_req_path(w, 10))
    torch.save(req, D._eval_req_path(w, 20))
    assert D._eval_requests(w) == [10, 20]
    X = 2
    assert not D._eval_shards_ready(w, 10, X)
    shards = []
    for r in range(X):
        texts = {st["name"]: {} for st in req["sets"]}
        for ch in D._eval_plan(req, r, X, chunk_seqs=512):
            for i in ch["rows"]:
                texts[ch["set"]][i] = [f"{ch['set']}-{i}-{j}" for j in range(ch["n"])]
        sh = {"ckpt_step": 10, "adapter_step": 11, "rank": r, "texts": texts, "t_gen": 1.0 + r, "n_seq": 0, "error": None}
        torch.save(sh, D._eval_shard_path(w, 10, r)); shards.append(sh)
    assert D._eval_shards_ready(w, 10, X) and not D._eval_shards_ready(w, 20, X)
    merged = D._eval_merge_shards([torch.load(D._eval_shard_path(w, 10, r), weights_only=False) for r in range(X)])
    for st in req["sets"]:
        assert sorted(merged[st["name"]]) == list(range(st["n_rows"]))
        assert all(len(v) == st["n"] for v in merged[st["name"]].values())
    assert merged["extra"][3] == ["extra-3-0", "extra-3-1", "extra-3-2"]


def test_eval_sets_from_assets_matches_rl_py_protocol():
    EU = types.SimpleNamespace(GEN_SEED=1234)
    es = {"realact_dirs": torch.randn(7, 8) * 3, "jlens_dirs": torch.randn(7, 8), "sae_dirs": torch.randn(5, 8)}
    EV = {"EU": EU, "es": es, "fams": ["realact", "jlens"], "feats": [1, 2, 3, 4, 5]}
    a = types.SimpleNamespace(eval_bo=4, eval_temp=1.0, eval_min_new=16, eval_max_new=64, max_new_tokens=96)
    sets = D._eval_sets_from_assets(EV, None, a)
    assert [s["name"] for s in sets] == ["realact", "jlens", "sae"]
    assert all(torch.allclose(s["dirs"].norm(dim=-1), torch.ones(s["n_rows"]), atol=1e-5) for s in sets)
    assert sets[0]["seed_base"] == 1234 * 1000 and sets[2]["feats"] == [1, 2, 3, 4, 5] and sets[0]["n"] == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("ALL OK")
