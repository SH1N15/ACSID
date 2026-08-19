"""Collision analysis across the ACSID SID variants.

For each mode's index.json we report:
  - Collision Rate = (N - #unique SIDs) / N
  - Unique SID Ratio = #unique SIDs / N
  - per-level collision: how many distinct tokens are used at each of the
    three quantizer levels (diagnostic: tells you whether collisions are
    concentrated at level 1, which is expected, or whether de-collision
    mostly happens at the leaf level).

Outputs a JSON file (default experiments/results/collision.json) and prints
a comparison table. Run from MiniOneRec/rq/ so ../data/Amazon and
../../experiments resolve as expected.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

# Make `from acsid...` importable when run as a plain script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def analyze_index(index: dict) -> dict:
    """index: {str(item_id): ["<a_N>","<b_N>","<c_N>"]}. Compute collision stats."""
    keys = list(index.keys())
    n = len(keys)

    # full SID string keys (concatenated tokens) == what convert_dataset writes
    full_sids = ["".join(v) for v in index.values()]
    unique_full = set(full_sids)
    collision_rate = (n - len(unique_full)) / n if n else 0.0
    unique_ratio = len(unique_full) / n if n else 0.0

    # per-level: distinct token strings at each position
    levels = max((len(v) for v in index.values()), default=0)
    per_level = []
    for lvl in range(levels):
        toks = [v[lvl] for v in index.values() if len(v) > lvl]
        cnt = Counter(toks)
        distinct = len(cnt)
        per_level.append({
            "level": lvl,
            "prefix": ["a", "b", "c", "d", "e"][lvl] if lvl < 5 else str(lvl),
            "distinct_tokens": distinct,
            "codebook_size": None,  # filled if known
            "max_collisions_at_level": max(cnt.values()) if cnt else 0,
        })

    return {
        "n_items": n,
        "unique_sids": len(unique_full),
        "collision_rate": collision_rate,
        "unique_ratio": unique_ratio,
        "per_level": per_level,
    }


def _load_index(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_args():
    import argparse

    ap = argparse.ArgumentParser(description="Collision analysis across ACSID SID variants")
    ap.add_argument("--base", default="../data/Amazon", help="data root (relative to cwd)")
    ap.add_argument("--dataset", default="Industrial_and_Scientific")
    ap.add_argument("--modes", nargs="+", default=["text", "fixed", "adaptive"])
    ap.add_argument("--include_upstream", action="store_true",
                    help="also analyze the shipped <dataset>.index.json as 'upstream' reference")
    ap.add_argument("--out_json", default=None,
                    help="output JSON path (default: ../../experiments/results/collision.json)")
    return ap.parse_args()


def main():
    args = parse_args()

    index_dir = os.path.join(args.base, "index")
    results = {}

    targets = [(m, os.path.join(index_dir, f"{args.dataset}.index.{m}.json")) for m in args.modes]
    if args.include_upstream:
        targets.insert(0, ("upstream", os.path.join(index_dir, f"{args.dataset}.index.json")))

    for mode, path in targets:
        if not os.path.exists(path):
            print(f"[analyze] MISSING: {mode} -> {path}, skipping")
            continue
        idx = _load_index(path)
        stats = analyze_index(idx)
        results[mode] = {"path": path, **stats}
        print(f"[analyze] {mode:10s}: N={stats['n_items']:5d} unique={stats['unique_sids']:5d} "
              f"collision_rate={stats['collision_rate']:.4f} unique_ratio={stats['unique_ratio']:.4f}")

    # comparison table
    print("\n===== Collision Rate Comparison =====")
    print(f"{'mode':12s} {'N':>6s} {'unique':>8s} {'coll_rate':>10s} {'uniq_ratio':>11s}")
    for mode, st in results.items():
        print(f"{mode:12s} {st['n_items']:6d} {st['unique_sids']:8d} "
              f"{st['collision_rate']:10.4f} {st['unique_ratio']:11.4f}")

    if len(results) >= 2:
        order = [m for m in ["upstream", "text", "fixed", "adaptive"] if m in results]
        print("\nordering (lower collision rate is better):")
        for m in order:
            print(f"  {m:12s} {results[m]['collision_rate']:.4f}")
        # directional check: adaptive <= fixed <= text (trend expected by PROJECT_PLAN §3.5)
        if {"text", "fixed", "adaptive"}.issubset(results.keys()):
            t, f, a = results["text"]["collision_rate"], results["fixed"]["collision_rate"], results["adaptive"]["collision_rate"]
            print(f"\n[check] adaptive({a:.4f}) <= fixed({f:.4f}) <= text({t:.4f}) ? "
                  f"{a <= f <= t}")

    out_json = args.out_json or os.path.join(_PROJECT_ROOT, "experiments", "results", "collision.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[analyze] wrote {out_json}")


if __name__ == "__main__":
    main()
