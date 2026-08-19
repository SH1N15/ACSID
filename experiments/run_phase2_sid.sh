#!/usr/bin/env bash
# Phase 2 — ACSID SID construction + collision analysis.
#
# Runs entirely on the cloud A10. Invokes the pipeline from MiniOneRec/rq/
# (same cwd convention as rqvae.sh).
#
#   item2vec -> cf.npy
#   compute_alpha -> alpha.npy
#   RQ-VAE (+P) training for modes: text, fixed, adaptive
#   generate_indices -> <dataset>.index.<mode>.json
#   regenerate_csv_sid -> data/Amazon/<mode>/{train,valid,test,info}
#   analyze_collision -> experiments/results/collision.json
#
# Usage:
#   bash experiments/run_phase2_sid.sh [DATASET]
#   DATASET default: Industrial_and_Scientific
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET="${1:-Industrial_and_Scientific}"

cd "${PROJECT_ROOT}/MiniOneRec/rq"
echo "[phase2] cwd=$(pwd)"
echo "[phase2] dataset=${DATASET}"

TEXT_PATH="../data/Amazon/index/${DATASET}.emb-qwen-td.npy"
if [[ ! -f "${TEXT_PATH}" ]]; then
    echo "[phase2] ERROR: text embeddings not found at ${TEXT_PATH}" >&2
    exit 1
fi

# Full SID-construction + CSV regeneration. Tune epochs/batch to your A10.
python "../../acsid/generate_sid.py" \
    --dataset "${DATASET}" \
    --text_path "${TEXT_PATH}" \
    --out_root "../data/Amazon" \
    --epochs 10000 \
    --batch_size 20480 \
    --eval_step 50 \
    --device cuda:0 \
    --modes text fixed adaptive

# Collision analysis across the three variants (+ shipped upstream SIDs).
python "../../acsid/analyze_collision.py" \
    --base "../data/Amazon" \
    --dataset "${DATASET}" \
    --include_upstream \
    --out_json "../../experiments/results/collision.json"

echo "[phase2] DONE. See experiments/results/collision.json"
