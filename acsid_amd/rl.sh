#!/bin/bash
# ACSID GRPO — AMD MI300X 192GB single GPU.
# Self-locating: bash acsid_amd/rl.sh  (invoke from anywhere).
# No accelerate, no DeepSpeed. num_generations=16 (NOT reduced), beam_search.
# RL optimizer: adamw_torch (bitsandbytes not supported on AMD).

set -euo pipefail
export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}/MiniOneRec"

MODEL_PATH="${MODEL_PATH:-path_to_sft_checkpoint}"

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    python ../acsid_amd/rl.py \
            --model_path "${MODEL_PATH}" \
            --train_batch_size 128 \
            --eval_batch_size 128 \
            --num_train_epochs 1 \
            --gradient_accumulation_steps 1 \
            --train_file "${train_file}" \
            --eval_file "${eval_file}" \
            --info_file "${info_file}" \
            --category ${category} \
            --sample_train False \
            --eval_step 0.0999 \
            --reward_type ranking \
            --num_generations 16 \
            --mask_all_zero False \
            --dynamic_sampling False \
            --sync_ref_model True \
            --beam_search True \
            --test_during_training False \
            --temperature 1.0 \
            --learning_rate 1e-5 \
            --add_gt False \
            --beta 1e-3 \
            --dapo False \
            --output_dir output_dir \
            --wandb_run_name grpo_text_seed42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.text.json \
            --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
