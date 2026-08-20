"""Rewrite only the SID columns of existing MiniOneRec CSVs.

Avoids re-running convert_dataset.py (which needs the .inter files we don't
have). Given the upstream train/valid/test CSVs + a new index.json, this
produces new CSVs identical to the originals except that history_item_sid and
item_sid are remapped to the new SIDs. Titles, IDs, user_ids, and the para
list-repr storage format are preserved exactly so downstream data.py reads
them unchanged. Also regenerates info/<category>_*.txt as sid\\ttitle\\tid,
matching the upstream contract.

The upstream CSVs store list cells as pandas repr; we eval them the same way
data.py does (SidSFTDataset.get_history uses eval(history_item_sid)).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import pandas as pd


def _sid_str(tokens) -> str:
    # contract: index.json values are ["<a_N>","<b_N>","<c_N>"]; concatenate
    return "".join(tokens)


def _safe_eval_list(cell):
    """Parse a repr string of a Python list (history_item_id / *_sid)."""
    if cell is None:
        return []
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        s = cell.strip()
        if s in ("", "[]"):
            return []
        try:
            return eval(s)
        except Exception:
            return []
    return []


def _find_single(glob_pattern: str, what: str) -> str:
    hits = sorted(glob.glob(glob_pattern))
    if len(hits) == 0:
        raise FileNotFoundError(f"no {what} matched {glob_pattern}")
    if len(hits) > 1:
        raise RuntimeError(f"expected exactly one {what} under {glob_pattern}, got {hits}")
    return hits[0]


def regenerate_one_csv(
    in_csv: str,
    out_csv: str,
    index: dict,
    items: dict,
) -> dict:
    """Rewrite one split CSV. Returns a small stats dict."""
    df = pd.read_csv(in_csv, dtype=object)  # keep list repr as raw strings

    n_rows = len(df)
    n_target_missing = 0
    n_hist_missing = 0

    new_hist_sid = []
    new_item_sid = []
    for _, row in df.iterrows():
        hist_ids = [int(x) for x in _safe_eval_list(row["history_item_id"])]
        target_id = int(row["item_id"])

        # history SIDs: keep list aligned with history_item_id; missing item
        # -> placeholder so lengths stay consistent (downstream consumes the
        # SID list as-is); we track counts for reporting.
        h_sids = []
        for iid in hist_ids:
            tok = index.get(str(iid))
            if tok is None:
                n_hist_missing += 1
                h_sids.append("")
            else:
                h_sids.append(_sid_str(tok))

        t_tok = index.get(str(target_id))
        if t_tok is None:
            n_target_missing += 1
            new_hist_sid.append(h_sids)
            new_item_sid.append("")
        else:
            new_hist_sid.append(h_sids)
            new_item_sid.append(_sid_str(t_tok))

    df["history_item_sid"] = new_hist_sid
    df["item_sid"] = new_item_sid

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return {
        "rows": n_rows,
        "target_missing": n_target_missing,
        "hist_missing": n_hist_missing,
        "out_csv": out_csv,
    }


def regenerate_info(
    out_info_txt: str,
    index: dict,
    items: dict,
) -> int:
    """Rewrite info/<category>_*.txt as sid\\ttitle\\tid for every item_id."""
    os.makedirs(os.path.dirname(os.path.abspath(out_info_txt)) or ".", exist_ok=True)
    n = 0
    with open(out_info_txt, "w", encoding="utf-8") as f:
        for item_id in sorted(index.keys(), key=lambda k: int(k)):
            tok = index[item_id]
            sid = _sid_str(tok)
            meta = items.get(str(item_id), {})
            title = meta.get("title", f"Item_{item_id}")
            # title may itself contain tabs/newlines; sanitize to one line
            title = (title or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
            if not title.strip():
                title = f"Item_{item_id}"
            f.write(f"{sid}\t{title}\t{item_id}\n")
            n += 1
    return n


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Rewrite SID columns of existing CSVs + regenerate info/*.txt (no .inter needed)")
    ap.add_argument("--src_dir", required=True,
                    help="upstream data dir containing train/ valid/ test/ info/ and {dataset}.item.json")
    ap.add_argument("--new_index", required=True, help="path to the new index.json")
    ap.add_argument("--out_dir", required=True, help="output data root (train/ valid/ test/ info/ created here)")
    ap.add_argument("--item_json", default=None,
                    help="path to item.json (default: {src_dir}/{item_basename}); used for info title column")
    ap.add_argument("--dataset", default="Industrial_and_Scientific",
                    help="dataset name used to locate item.json and CSV/info basenames")
    args = ap.parse_args()

    src = args.src_dir
    # locate source CSV/info by glob so we inherit the exact basenames;
    # filter by dataset name prefix to avoid matching Office_Products etc.
    src_train = _find_single(os.path.join(src, "train", f"{args.dataset}*11.csv"), "train csv")
    src_valid = _find_single(os.path.join(src, "valid", f"{args.dataset}*11.csv"), "valid csv")
    src_test = _find_single(os.path.join(src, "test", f"{args.dataset}*11.csv"), "test csv")
    src_info = _find_single(os.path.join(src, "info", f"{args.dataset}*11.txt"), "info txt")
    csv_basename = os.path.basename(src_train)
    info_basename = os.path.basename(src_info)

    item_json = args.item_json or os.path.join(src, f"{args.dataset}.item.json")
    with open(args.new_index, "r") as f:
        index = json.load(f)
    with open(item_json, "r") as f:
        items = json.load(f)
    print(f"[regen] index items: {len(index)}; item.json items: {len(items)}; "
          f"csv_base={csv_basename}; info_base={info_basename}")

    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "valid"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "info"), exist_ok=True)

    for split_src, split_name in [(src_train, "train"), (src_valid, "valid"), (src_test, "test")]:
        out_csv = os.path.join(out_dir, split_name, csv_basename)
        st = regenerate_one_csv(split_src, out_csv, index, items)
        print(f"[regen] {split_name}: {st}")

    out_info = os.path.join(out_dir, "info", info_basename)
    n = regenerate_info(out_info, index, items)
    print(f"[regen] info: wrote {n} lines -> {out_info}")


if __name__ == "__main__":
    main()
