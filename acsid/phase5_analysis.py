"""Phase 5 analysis: popularity-stratified metrics + SID collision case study.

Inputs (paths relative to MiniOneRec/, overridable via CLI):
  - results/eval_{sft,grpo}_{mode}_seed42.json   per-sample predictions
                                                 (predict/output are SID strings
                                                 in THAT mode's SID space)
  - data/Amazon/text/train/*.csv                 item popularity (freq; identical
                                                 across modes, only SID cols differ)
  - data/Amazon/index/{DS}.index.{mode}.json     per-mode SID tables (SID -> item id)
  - data/Amazon/index/{DS}.item.json             optional item titles (case study)

Buckets (by TARGET item's train frequency):
  cold  freq == 0   |  low / mid / high = tertiles of nonzero freq

Metrics replicate calc.py exactly (HR@K; NDCG@K = 1/log2(rank+2), rank 0-based)
and the script VALIDATES itself by printing overall numbers next to the known
calc.py outputs before any stratified table is trusted.

Run:  cd MiniOneRec && python ../acsid/phase5_analysis.py
Writes: ../experiments/results/phase5_analysis.json
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

def compute_item_freq(train_csv: str) -> "np.ndarray":
    """Mirror of adaptive_fusion.compute_item_freq (pandas only, no torch import
    so the analysis runs on GPU-less boxes)."""
    import pandas as pd

    df = pd.read_csv(train_csv)
    max_id = -1
    counts: dict[int, int] = {}

    def _bump(item_ids) -> None:
        nonlocal max_id
        for x in item_ids:
            x = int(x)
            counts[x] = counts.get(x, 0) + 1
            if x > max_id:
                max_id = x

    for hist, tgt in zip(df["history_item_id"], df["item_id"]):
        try:
            history_list = eval(hist) if isinstance(hist, str) else hist  # noqa: S307
        except Exception:
            history_list = []
        if history_list is None:
            history_list = []
        _bump(history_list)
        _bump([int(tgt)])

    arr = np.zeros(max(max_id + 1, 0), dtype=np.int64)
    for k, v in counts.items():
        arr[k] = v
    return arr

DATASET = "Industrial_and_Scientific"
ALPHA_MAX = 0.3
KS = [5, 10]

# known calc.py outputs (seed 42) -- self-validation gate
REFERENCE = {
    "sft_text":    0.0734,
    "sft_fixed":   0.0952,
    "sft_adaptive": 0.0850,
    "grpo_text":   0.0986,
    "grpo_adaptive": 0.1053,
}

EVAL_FILES = {
    "sft_text": "text",
    "sft_fixed": "fixed",
    "sft_adaptive": "adaptive",
    "grpo_text": "text",
    "grpo_adaptive": "adaptive",
}
EVAL_PATHS = {
    "sft_text": "results/eval_sft_text_seed42.json",
    "sft_fixed": "results/eval_sft_fixed_seed42.json",
    "sft_adaptive": "results/eval_sft_adaptive_seed42.json",
    "grpo_text": "results/eval_grpo_text_seed42.json",
    "grpo_adaptive": "results/eval_grpo_adaptive_seed42.json",
}


def make_buckets(freq: np.ndarray) -> tuple[dict[int, str], dict[str, float]]:
    """item id -> bucket label; plus per-bucket mean alpha (adaptive formula)."""
    n_ref = float(np.median(freq[freq > 0])) if (freq > 0).any() else 1.0
    alpha = ALPHA_MAX * np.minimum(1.0, np.log1p(freq) / math.log1p(n_ref))
    alpha[freq <= 0] = 0.0

    nonzero = sorted(np.nonzero(freq)[0], key=lambda i: freq[i])
    n = len(nonzero)
    bucket_of = {int(i): "cold" for i in range(len(freq))}
    for rank, i in enumerate(nonzero):
        bucket_of[int(i)] = "low" if rank < n / 3 else ("mid" if rank < 2 * n / 3 else "high")

    mean_alpha = {}
    for b in ["cold", "low", "mid", "high"]:
        ids = [i for i, bb in bucket_of.items() if bb == b]
        mean_alpha[b] = float(np.mean(alpha[ids])) if ids else 0.0
    return bucket_of, mean_alpha


def target_sid(sample: dict) -> str:
    out = sample["output"]
    t = out[0] if isinstance(out, list) else out
    return t.strip(' \n"')


def rank_of_target(sample: dict) -> int | None:
    """First index (0-based) of the ground truth in the prediction list, or None.

    Matching is pure SID-string equality within the mode's own SID space
    (exactly what calc.py does) -- independent of item-id resolution.
    """
    t = target_sid(sample)
    for i, p in enumerate(sample["predict"]):
        if p.strip('"\n').strip() == t:
            return i
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"n": n}
    for k in KS:
        hits = [r for r in ranks if r is not None and r < k]
        out[f"HR@{k}"] = len(hits) / n if n else 0.0
        out[f"NDCG@{k}"] = sum(1.0 / math.log2(r + 2) for r in hits) / n if n else 0.0
    return out


def main() -> None:
    # per-mode train CSV preferred; upstream top-level is identical for item
    # frequencies (only SID columns differ between modes)
    train_csv = sorted(glob.glob("data/Amazon/text/train/*.csv")
                       or glob.glob("data/Amazon/train/*.csv"))[0]
    freq = compute_item_freq(train_csv)
    bucket_of, mean_alpha = make_buckets(freq)

    # per-mode reverse index: SID string -> [item ids] (a SID shared by >1 item
    # = collision; such targets are excluded from buckets, counted separately)
    rev_index = {}
    titles = {}
    item_json = f"data/Amazon/index/{DATASET}.item.json"
    if os.path.exists(item_json):
        with open(item_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        titles = {int(k): (v if isinstance(v, str) else str(v)) for k, v in raw.items()}
    for mode in ["text", "fixed", "adaptive"]:
        with open(f"data/Amazon/index/{DATASET}.index.{mode}.json", "r", encoding="utf-8") as f:
            idx = json.load(f)
        rev = defaultdict(list)
        for iid, toks in idx.items():
            rev["".join(toks)].append(int(iid))
        rev_index[mode] = dict(rev)

    # cross-mode common subset: exclude any target item whose SID collides in
    # ANY mode, so every bucket compares the SAME samples across modes.
    # (text alone has 11 collision groups ~= 260 popular test targets that are
    # easy hits for text -- keeping them in only some modes' buckets inflates
    # cross-mode deltas.)
    ambiguous_ids = set()
    for mode in rev_index:
        for ids in rev_index[mode].values():
            if len(ids) > 1:
                ambiguous_ids.update(ids)
    print(f"collision-ambiguous items (any mode, excluded from ALL modes' buckets): {len(ambiguous_ids)}")

    print(f"items={len(freq)}  cold_items={(freq == 0).sum()}  "
          f"n_ref(median nonzero)={np.median(freq[freq > 0]):.0f}")
    print(f"mean adaptive alpha per bucket: "
          + "  ".join(f"{b}={mean_alpha[b]:.3f}" for b in ["cold", "low", "mid", "high"]))

    # ---- stratified metrics -------------------------------------------------
    strat = {}
    mismatches = []
    print("\n===== SELF-VALIDATION (overall NDCG@10 must match calc.py) =====")
    for key, mode in EVAL_FILES.items():
        with open(EVAL_PATHS[key], "r", encoding="utf-8") as f:
            data = json.load(f)
        ranks, per_bucket, common_ranks = [], defaultdict(list), []
        excluded = 0
        for s in data:
            r = rank_of_target(s)
            ranks.append(r)
            ids = rev_index[mode].get(target_sid(s))
            if ids is None or len(ids) > 1 or ids[0] in ambiguous_ids:
                excluded += 1
                continue
            per_bucket[bucket_of.get(ids[0], "cold")].append(r)
            common_ranks.append(r)
        overall = metrics(ranks)
        common_all = metrics(common_ranks)
        ref = REFERENCE[key]
        flag = "OK" if abs(overall["NDCG@10"] - ref) < 0.002 else "MISMATCH!"
        if flag != "OK":
            mismatches.append(key)
        print(f"{key:14s} NDCG@10={overall['NDCG@10']:.4f}  (calc.py {ref})  {flag}"
              f"  common_n={common_all['n']} (excluded {excluded})")
        strat[key] = {b: metrics(per_bucket[b]) for b in ["cold", "low", "mid", "high"]}
        strat[key]["ALL"] = overall
        strat[key]["COMMON"] = common_all
    if mismatches:
        print(f"\n*** VALIDATION FAILED for {mismatches} -- stratified tables below are NOT trustworthy ***")
        sys.exit(1)

    for metric in ["NDCG@10", "HR@10", "NDCG@5", "HR@5"]:
        print(f"\n===== {metric} by target-popularity bucket (common subset) =====")
        print(f"{'bucket':8s}" + "".join(f"{k:>15s}" for k in EVAL_FILES))
        for b in ["cold", "low", "mid", "high", "COMMON", "ALL"]:
            row = f"{b:8s}"
            for k in EVAL_FILES:
                row += f"{strat[k][b][metric]:15.4f}"
            if b not in ("ALL",):
                row += f"   (n={strat['sft_text'][b]['n']})"
            else:
                row += "   (n=4533, raw)"
            print(row)

    print("\n===== deltas: adaptive - text (NDCG@10, common subset) =====")
    for stage in ["sft", "grpo"]:
        for b in ["cold", "low", "mid", "high", "COMMON"]:
            d = strat[f"{stage}_adaptive"][b]["NDCG@10"] - strat[f"{stage}_text"][b]["NDCG@10"]
            print(f"{stage:5s} {b:8s} {d:+.4f}")

    print("\n===== deltas: GRPO - SFT within mode (NDCG@10, common subset) =====")
    for mode in ["text", "adaptive"]:
        for b in ["low", "mid", "high", "COMMON"]:
            d = strat[f"grpo_{mode}"][b]["NDCG@10"] - strat[f"sft_{mode}"][b]["NDCG@10"]
            print(f"{mode:9s} {b:8s} {d:+.4f}")

    # ---- collision case study ----------------------------------------------
    print("\n===== case study: text-collision groups vs adaptive/fixed =====")
    index = {m: {int(i): t for i, t in json.load(
        open(f"data/Amazon/index/{DATASET}.index.{m}.json", encoding="utf-8")).items()}
        for m in ["text", "fixed", "adaptive"]}

    text_groups = defaultdict(list)
    for iid, toks in index["text"].items():
        text_groups["".join(toks)].append(iid)
    colliding = {sid: ids for sid, ids in text_groups.items() if len(ids) > 1}
    sep_adaptive = sep_fixed = 0
    for sid, ids in colliding.items():
        if len({"".join(index["adaptive"][i]) for i in ids}) == len(ids):
            sep_adaptive += 1
        if len({"".join(index["fixed"][i]) for i in ids}) == len(ids):
            sep_fixed += 1
    n_groups = len(colliding)
    n_items_in_groups = sum(len(v) for v in colliding.values())
    print(f"text collision groups: {n_groups} (covering {n_items_in_groups} items) | "
          f"fully separated in adaptive: {sep_adaptive}/{n_groups}, in fixed: {sep_fixed}/{n_groups}")

    def title(i: int) -> str:
        return (titles.get(i) or f"item{i}")[:45]

    for sid, ids in list(colliding.items())[:8]:
        print(f"- textSID={sid}")
        for i in ids:
            print(f"    {i:5d} {title(i):47s} adaptiveSID={''.join(index['adaptive'][i])[:24]}")

    out = {
        "buckets": {"mean_alpha": mean_alpha},
        "stratified": strat,
        "case_study": {
            "text_collision_groups": n_groups,
            "separated_in_adaptive": sep_adaptive,
            "separated_in_fixed": sep_fixed,
        },
    }
    out_path = os.path.join(_PROJECT_ROOT, "experiments", "results", "phase5_analysis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
