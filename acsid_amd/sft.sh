#!/bin/bash
# ACSID SFT — AMD MI300X 192GB single GPU (full-param, bf16, no quantization).
# Self-locating: bash acsid_amd/sft.sh  (invoke from anywhere).
# It cd's into MiniOneRec/ so ./data/Amazon/... resolves, then runs
#   python ../acsid_amd/sft.py  (the AMD-adapted entrypoint).
# micro_batch 16 x accum 64 = 1024 effective batch (parity with the original
# 8-GPU config).
#
# Targets the ACSID text-mode SID outputs (index.text.json + text/ CSVs).
# For a Phase-1 baseline reproducing the OFFICIAL SIDs, swap --sid_index_path
# and the train/eval CSVs to the upstream-shipped ./data/Amazon/index/*.index.json
# and ./data/Amazon/{train,valid}/*.csv.

set -euo pipefail
export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}/MiniOneRec"

BASE_MODEL="${BASE_MODEL:-/path/to/Qwen2.5-3B-Base}"

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}

    python ../acsid_amd/sft.py \
            --base_model "${BASE_MODEL}" \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file "${train_file}" \
            --eval_file "${eval_file}" \
            --output_dir output_dir/sft_text_seed42 \
            --wandb_project "" \
            --wandb_run_name sft_text_seed42 \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.text.json \
            --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
            --freeze_LLM False
done
