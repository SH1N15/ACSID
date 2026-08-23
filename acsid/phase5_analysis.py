"""Phase 5 analysis: popularity-stratified metrics + SID collision case study.

Inputs (paths relative to MiniOneRec/, overridable via CLI):
  - results/eval_{sft,grpo}_{mode}_seed42.json   per-sample predictions
  - data/Amazon/{mode}/info/*.txt                name <-> item id (line position)
  - data/Amazon/text/train/*.csv                 item popularity (freq; identical
                                                 across modes, only SID cols differ)
  - data/Amazon/index/{DS}.index.{mode}.json     per-mode SID tables

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

from adaptive_fusion import compute_item_freq  # noqa: E402  (same package dir)

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
    "sft_text": "results/eval_sft_text_seed42.json",
    "sft_fixed": "results/eval_sft_fixed_seed42.json",
    "sft_adaptive": "results/eval_sft_adaptive_seed42.json",
    "grpo_text": "results/eval_grpo_text_seed42.json",
    "grpo_adaptive": "results/eval_grpo_adaptive_seed42.json",
}


def load_info(mode: str) -> tuple[dict[int, str], dict[str, list[int]]]:
    path = glob.glob(f"data/Amazon/{mode}/info/{DATASET}*.txt")[0]
    id2name, name2ids = {}, defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            name = line.split("\t")[0].strip()
            id2name[i] = name
            name2ids[name].append(i)
    return id2name, name2ids


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


def rank_of_target(sample: dict) -> int | None:
    """First index (0-based) of the ground truth in the prediction list, or None."""
    preds = [p.strip('"\n').strip() for p in sample["predict"]]
    out = sample["output"]
    target = out[0].strip('"').strip() if isinstance(out, list) else out.strip(' \n"')
    for i, p in enumerate(preds):
        if p == target:
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
    train_csv = sorted(glob.glob("data/Amazon/text/train/*.csv"))[0]
    freq = compute_item_freq(train_csv)
    id2name, name2ids = load_info("text")
    bucket_of, mean_alpha = make_buckets(freq)
    dup_names = sum(1 for ids in name2ids.values() if len(ids) > 1)

    print(f"items={len(freq)}  cold_items={(freq == 0).sum()}  "
          f"n_ref(median nonzero)={np.median(freq[freq > 0]):.0f}  dup_names={dup_names}")
    print(f"mean adaptive alpha per bucket: "
          + "  ".join(f"{b}={mean_alpha[b]:.3f}" for b in ["cold", "low", "mid", "high"]))

    # ---- stratified metrics -------------------------------------------------
    strat = {}
    print("\n===== SELF-VALIDATION (overall NDCG@10 must match calc.py) =====")
    for key, path in EVAL_FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranks, per_bucket = [], defaultdict(list)
        unmatched = 0
        for s in data:
            out = s["output"]
            tname = out[0] if isinstance(out, list) else out
            tname = tname.strip(' \n"')
            ids = name2ids.get(tname)
            if not ids:
                unmatched += 1
                ranks.append(None)
                continue
            r = rank_of_target(s)
            ranks.append(r)
            per_bucket[bucket_of.get(ids[0], "cold")].append(r)
        overall = metrics(ranks)
        ref = REFERENCE[key]
        flag = "OK" if abs(overall["NDCG@10"] - ref) < 0.002 else "MISMATCH!"
        print(f"{key:14s} NDCG@10={overall['NDCG@10']:.4f}  (calc.py {ref})  {flag}"
              + (f"  unmatched={unmatched}" if unmatched else ""))
        strat[key] = {b: metrics(per_bucket[b]) for b in ["cold", "low", "mid", "high"]}
        strat[key]["ALL"] = overall

    for metric in ["NDCG@10", "HR@10", "NDCG@5", "HR@5"]:
        print(f"\n===== {metric} by target-popularity bucket =====")
        hdr = f"{'bucket':8s}" + "".join(f"{k:>15s}" for k in EVAL_FILES)
        print(hdr)
        for b in ["cold", "low", "mid", "high", "ALL"]:
            row = f"{b:8s}"
            for k in EVAL_FILES:
                v = strat[k][b][metric]
                row += f"{v:15.4f}"
            if b != "ALL":
                n = strat["sft_text"][b]["n"]
                row += f"   (n={n})"
            print(row)

    print("\n===== deltas: adaptive - text (NDCG@10) =====")
    for stage in ["sft", "grpo"]:
        for b in ["cold", "low", "mid", "high", "ALL"]:
            d = strat[f"{stage}_adaptive"][b]["NDCG@10"] - strat[f"{stage}_text"][b]["NDCG@10"]
            print(f"{stage:5s} {b:6s} {d:+.4f}")

    # ---- collision case study ----------------------------------------------
    print("\n===== case study: text-collision groups vs adaptive/fixed =====")
    idx_dir = "data/Amazon/index"
    index = {}
    for m in ["text", "fixed", "adaptive"]:
        with open(f"{idx_dir}/{DATASET}.index.{m}.json", "r", encoding="utf-8") as f:
            index[m] = json.load(f)

    text_groups = defaultdict(list)
    for iid, toks in index["text"].items():
        text_groups["".join(toks)].append(int(iid))
    colliding = {sid: ids for sid, ids in text_groups.items() if len(ids) > 1}
    sep_adaptive = sep_fixed = 0
    for sid, ids in colliding.items():
        if len({"".join(index["adaptive"][str(i)]) for i in ids}) == len(ids):
            sep_adaptive += 1
        if len({"".join(index["fixed"][str(i)]) for i in ids}) == len(ids):
            sep_fixed += 1
    n_groups = len(colliding)
    n_items_in_groups = sum(len(v) for v in colliding.values())
    print(f"text collision groups: {n_groups} (covering {n_items_in_groups} items) | "
          f"fully separated in adaptive: {sep_adaptive}/{n_groups}, in fixed: {sep_fixed}/{n_groups}")

    for sid, ids in list(colliding.items())[:8]:
        names = [id2name.get(i, "?")[:45] for i in ids]
        print(f"- ids={ids} textSID={sid[:30]}...")
        for i, nm in zip(ids, names):
            print(f"    {i:5d} {nm:47s} adaptiveSID={''.join(index['adaptive'][str(i)])[:24]}")

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
