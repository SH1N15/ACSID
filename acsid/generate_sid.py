"""ACSID end-to-end orchestrator (run on the cloud A10).

Assumes cwd = MiniOneRec/rq/ (same convention as rqvae.sh), so all relative
paths resolve against that directory:

    text embeddings : ../data/Amazon/index/<dataset>.emb-qwen-td.npy
    train CSV       : ../data/Amazon/train/<dataset>_5_2016-10-2018-11.csv
    cf.npy / alpha  : ../data/Amazon/index/cf.npy , alpha.npy
    RQ-VAE ckpts    : ./output/<dataset>/<mode>/<timestamp>/best_collision_model.pth
    index.<mode>.json : ../data/Amazon/index/<dataset>.index.<mode>.json
    regenerated data : ../data/Amazon/<mode>/{train,valid,test,info}/...

Pipelines per mode (shared training protocol — codebook, e_dim, layers, lr,
epochs, seed — only the input representation differs, per PROJECT_PLAN.md
§2.4):

    item2vec.py  -> cf.npy
    (compute alpha) -> alpha.npy        # adaptive only
    rqvae.py --mode {mode}             # trains RQ-VAE (+P jointly when mode!=text)
    generate_indices.py -> index.<mode>.json
    regenerate_csv_sid.py -> data/Amazon/<mode>/
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

# When run as `python ../../acsid/generate_sid.py` from rq/, this file's own
# dir is acsid/. We need the project root on sys.path so `from acsid...` works
# inside the helpers we call here.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
from acsid.adaptive_fusion import compute_item_freq, compute_alpha  # noqa: E402


def _run(cmd, cwd=None, env=None):
    print("[run] " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def _latest_best_collision(ckpt_root: str) -> str:
    pattern = os.path.join(ckpt_root, "*", "best_collision_model.pth")
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(f"no best_collision_model.pth under {pattern}")
    return hits[-1]


def ensure_item2vec(train_csv: str, cf_npy: str, cf_dim: int, epochs: int, seed: int, n_items=None):
    if os.path.exists(cf_npy):
        print(f"[generate_sid] cf.npy exists, skipping item2vec: {cf_npy}")
        return
    from acsid.item2vec import train_item2vec
    train_item2vec(
        train_csv=train_csv,
        output_npy=cf_npy,
        dim=cf_dim,
        epochs=epochs,
        seed=seed,
        n_items=n_items,
    )


def ensure_alpha(train_csv: str, alpha_npy: str, alpha_max: float):
    if os.path.exists(alpha_npy):
        print(f"[generate_sid] alpha.npy exists, skipping: {alpha_npy}")
        return
    freq = compute_item_freq(train_csv)
    alpha = compute_alpha(freq, alpha_max=alpha_max, fixed=False)
    os.makedirs(os.path.dirname(os.path.abspath(alpha_npy)) or ".", exist_ok=True)
    np.save(alpha_npy, alpha)
    nonzero = int((freq > 0).sum())
    print(f"[generate_sid] alpha.npy shape={alpha.shape}; "
          f"items_with_train_interactions={nonzero}; "
          f"alpha stats: min={alpha.min():.4f} max={alpha.max():.4f} mean={alpha.mean():.4f}")


def parse_args():
    import argparse

    ap = argparse.ArgumentParser(description="ACSID full pipeline: item2vec -> alpha -> 3x RQ-VAE -> index -> CSV")
    ap.add_argument("--dataset", default="Industrial_and_Scientific")
    ap.add_argument("--text_path", default=None,
                    help="default: ../data/Amazon/index/<dataset>.emb-qwen-td.npy")
    ap.add_argument("--train_csv", default=None,
                    help="default: ../data/Amazon/train/<dataset>_5_2016-10-2018-11.csv")
    ap.add_argument("--out_root", default="../data/Amazon", help="data root (relative to cwd=rq/)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=20480)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_step", type=int, default=50)
    ap.add_argument("--alpha_max", type=float, default=0.3)
    ap.add_argument("--cf_dim", type=int, default=256, help="Item2Vec dim")
    ap.add_argument("--item2vec_epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2024, help="RQ-VAE seed (matches rqvae.py default)")
    ap.add_argument("--modes", nargs="+", default=["text", "fixed", "adaptive"],
                    choices=["text", "fixed", "adaptive"])
    ap.add_argument("--skip_item2vec", action="store_true")
    ap.add_argument("--skip_alpha", action="store_true")
    ap.add_argument("--rqvae_py", default="rqvae.py", help="filename of the rqvae driver (relative to cwd)")
    ap.add_argument("--gen_indices_py", default="generate_indices.py")
    ap.add_argument("--regen_py", default=None,
                    help="default: <project_root>/acsid/regenerate_csv_sid.py")
    return ap.parse_args()


def main():
    args = parse_args()

    rq_dir = os.getcwd()
    out_root = os.path.abspath(args.out_root)
    index_dir = os.path.join(out_root, "index")

    text_path = args.text_path or os.path.join(index_dir, f"{args.dataset}.emb-qwen-td.npy")
    train_csv = args.train_csv or os.path.join(out_root, "train", f"{args.dataset}_5_2016-10-2018-11.csv")
    cf_npy = os.path.join(index_dir, "cf.npy")
    alpha_npy = os.path.join(index_dir, "alpha.npy")

    print(f"[generate_sid] cwd={rq_dir}")
    print(f"[generate_sid] text_path={text_path}")
    print(f"[generate_sid] train_csv={train_csv}")
    print(f"[generate_sid] cf_npy={cf_npy}  alpha_npy={alpha_npy}")
    print(f"[generate_sid] modes={args.modes}  alpha_max={args.alpha_max}")

    if not os.path.exists(text_path):
        raise FileNotFoundError(f"text embeddings not found: {text_path}")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train csv not found: {train_csv}")

    # 1) collaborative embedding (train-only, CPU)
    if args.skip_item2vec:
        print("[generate_sid] --skip_item2vec; assuming cf.npy is ready")
    else:
        ensure_item2vec(train_csv, cf_npy, args.cf_dim, args.item2vec_epochs, args.seed)

    # 2) per-item adaptive alpha (even when 'fixed' is in modes: fixed path
    #    does not read alpha_path, so this only needs to exist for 'adaptive')
    if args.skip_alpha:
        print("[generate_sid] --skip_alpha; assuming alpha.npy is ready")
    else:
        ensure_alpha(train_csv, alpha_npy, args.alpha_max)

    regen_py = args.regen_py or os.path.join(_PROJECT_ROOT, "acsid", "regenerate_csv_sid.py")
    if not os.path.isabs(regen_py):
        regen_py = os.path.join(_PROJECT_ROOT, "acsid", "regenerate_csv_sid.py")

    # 3) per-mode: train RQ-VAE (+P) -> indices -> regenerated CSV
    for mode in args.modes:
        print("\n" + "=" * 70)
        print(f"[generate_sid] MODE = {mode}")
        print("=" * 70)

        ckpt_dir = os.path.join(".", "output", args.dataset, mode)  # relative to cwd=rq/
        os.makedirs(ckpt_dir, exist_ok=True)
        index_out = os.path.join(index_dir, f"{args.dataset}.index.{mode}.json")

        rq_cmd = [
            sys.executable, args.rqvae_py,
            "--data_path", text_path,
            "--ckpt_dir", ckpt_dir,
            "--lr", str(args.lr),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--eval_step", str(args.eval_step),
            "--device", args.device,
            "--mode", mode,
            "--alpha_max", str(args.alpha_max),
        ]
        if mode != "text":
            rq_cmd += ["--cf_path", cf_npy]
        if mode == "adaptive":
            rq_cmd += ["--alpha_path", alpha_npy]
        _run(rq_cmd)

        best_ckpt = _latest_best_collision(ckpt_dir)
        print(f"[generate_sid] best_collision ckpt: {best_ckpt}")

        gen_cmd = [
            sys.executable, args.gen_indices_py,
            "--ckpt_path", best_ckpt,
            "--output_file", index_out,
            "--device", args.device,
        ]
        _run(gen_cmd)

        # 4) regenerate CSVs + info using the NEW index.json (src = upstream dir)
        item_json_path = os.path.join(index_dir, f"{args.dataset}.item.json")
        regen_cmd = [
            sys.executable, regen_py,
            "--src_dir", out_root,
            "--new_index", index_out,
            "--out_dir", os.path.join(out_root, mode),
            "--dataset", args.dataset,
            "--item_json", item_json_path,
        ]
        _run(regen_cmd)

        print(f"[generate_sid] DONE mode={mode} -> index={index_out} data={os.path.join(out_root, mode)}")

    print("\n[generate_sid] ALL MODES COMPLETE")


if __name__ == "__main__":
    main()
