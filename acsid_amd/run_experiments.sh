#!/bin/bash
# ACSID full experiment matrix — AMD MI300X 192GB single GPU
#
# 6 SFT runs (3 methods x 2 seeds) + 4 GRPO runs (2 methods x 2 seeds) = 10 total
#
# Usage:
#   bash acsid_amd/run_experiments.sh
#
# Prerequisites:
#   1. Phase 2 SID construction completed (3 sets of index.*.json + CSVs)
#   2. Qwen2.5-3B-Base downloaded and path set below
#   3. Run from MiniOneRec/ directory
set -euo pipefail

export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES=0

BASE_MODEL="/path/to/Qwen2.5-3B-Base"
DATASET="Industrial_and_Scientific"
ITEM_META="./data/Amazon/index/${DATASET}.item.json"

declare -A SID_INDEX=(
    ["text"]="./data/Amazon/index/${DATASET}.index.text.json"
    ["fixed"]="./data/Amazon/index/${DATASET}.index.fixed.json"
    ["adaptive"]="./data/Amazon/index/${DATASET}.index.adaptive.json"
)

declare -A DATA_DIR=(
    ["text"]="./data/Amazon/text"
    ["fixed"]="./data/Amazon/fixed"
    ["adaptive"]="./data/Amazon/adaptive"
)

SEEDS=(42 123)

# -------------------------------------------------------
# SFT: 3 methods x 2 seeds = 6 runs
# -------------------------------------------------------
echo "===== SFT PHASE ====="
for mode in text fixed adaptive; do
    for seed in "${SEEDS[@]}"; do
        echo "--- SFT: mode=${mode} seed=${seed} ---"
        train_file=$(ls -f ${DATA_DIR[$mode]}/train/${DATASET}*11.csv)
        eval_file=$(ls -f ${DATA_DIR[$mode]}/valid/${DATASET}*11.csv)
        test_file=$(ls -f ${DATA_DIR[$mode]}/test/${DATASET}*11.csv)
        info_file=$(ls -f ${DATA_DIR[$mode]}/info/${DATASET}*.txt)

        python sft.py \
            --base_model ${BASE_MODEL} \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/sft_${mode}_seed${seed} \
            --wandb_project "" \
            --wandb_run_name sft_${mode}_seed${seed} \
            --category ${DATASET} \
            --train_from_scratch False \
            --seed ${seed} \
            --sid_index_path ${SID_INDEX[$mode]} \
            --item_meta_path ${ITEM_META} \
            --freeze_LLM False
    done
done

# -------------------------------------------------------
# GRPO: 2 methods (text, adaptive) x 2 seeds = 4 runs
# -------------------------------------------------------
echo "===== GRPO PHASE ====="
for mode in text adaptive; do
    for seed in "${SEEDS[@]}"; do
        echo "--- GRPO: mode=${mode} seed=${seed} ---"
        sft_ckpt="output_dir/sft_${mode}_seed${seed}"
        train_file=$(ls -f ${DATA_DIR[$mode]}/train/${DATASET}*.csv)
        eval_file=$(ls -f ${DATA_DIR[$mode]}/valid/${DATASET}*11.csv)
        info_file=$(ls -f ${DATA_DIR[$mode]}/info/${DATASET}*.txt)

        python rl.py \
            --model_path ${sft_ckpt} \
            --train_batch_size 64 \
            --eval_batch_size 128 \
            --num_train_epochs 2 \
            --gradient_accumulation_steps 2 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --info_file ${info_file} \
            --category ${DATASET} \
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
            --output_dir output_dir/grpo_${mode}_seed${seed} \
            --wandb_run_name grpo_${mode}_seed${seed} \
            --sid_index_path ${SID_INDEX[$mode]} \
            --item_meta_path ${ITEM_META}
    done
done

echo "===== ALL DONE ====="