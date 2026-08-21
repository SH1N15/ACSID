#!/bin/bash
# ACSID full experiment matrix — AMD MI300X 192GB single GPU.
# 6 SFT (3 methods x 2 seeds) + 4 GRPO (2 methods x 2 seeds) = 10 runs.
#
# Self-locating: bash acsid_amd/run_experiments.sh  (invoke from anywhere).
# Runs from MiniOneRec/ so ./data/Amazon/... resolves; launches the AMD-adapted
# entries at ../acsid_amd/sft.py and ../acsid_amd/rl.py.
#
# Prereqs:
#   1. Phase 2 SID construction done (acsid/generate_sid.py produced
#      ./data/Amazon/index/<dataset>.index.{text,fixed,adaptive}.json and
#      ./data/Amazon/{text,fixed,adaptive}/{train,valid,test,info}/...).
#   2. Qwen2.5-3B-Base downloaded; set BASE_MODEL env var (absolute path).

set -euo pipefail
export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES=0
# wandb prompts for an API key in non-interactive (nohup) runs and crashes.
# Disable it; set WANDB_MODE=online and provide an API key if you want logging.
export WANDB_MODE="${WANDB_MODE:-disabled}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}/MiniOneRec"

BASE_MODEL="${BASE_MODEL:-/path/to/Qwen2.5-3B-Base}"
DATASET="Industrial_and_Scientific"
ITEM_META="./data/Amazon/index/${DATASET}.item.json"

# Seeds and phases are env-overridable so you can run a subset, e.g.:
#   PHASES="sft eval" SEEDS_STR="42" bash acsid_amd/run_experiments.sh
# Default: 3 SFT + 3 eval + 2 GRPO (seed=42 only; 100GB storage can't
# hold 6 SFT checkpoints at ~18GB each).
read -ra SEEDS <<< "${SEEDS_STR:-42}"
PHASES="${PHASES:-sft grpo}"

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

# -------------------------------------------------------
# SFT: 3 methods x N seeds
# -------------------------------------------------------
if [[ "${PHASES}" == *"sft"* ]]; then
echo "===== SFT PHASE ====="
# SKIP_MODES: space-separated modes to skip (e.g. "text" when already done).
# Example: SKIP_MODES="text" PHASES="sft" SEEDS_STR="42" bash acsid_amd/run_experiments.sh
SKIP_MODES="${SKIP_MODES:-}"
for mode in text fixed adaptive; do
    if [[ " ${SKIP_MODES} " == *" ${mode} "* ]]; then
        echo "--- SFT: skipping mode=${mode} (in SKIP_MODES) ---"
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        echo "--- SFT: mode=${mode} seed=${seed} ---"
        train_file=$(ls -f ${DATA_DIR[$mode]}/train/${DATASET}*11.csv)
        eval_file=$(ls -f ${DATA_DIR[$mode]}/valid/${DATASET}*11.csv)
        test_file=$(ls -f ${DATA_DIR[$mode]}/test/${DATASET}*11.csv)
        info_file=$(ls -f ${DATA_DIR[$mode]}/info/${DATASET}*.txt)

        python ../acsid_amd/sft.py \
            --base_model "${BASE_MODEL}" \
            --batch_size 1024 \
            --micro_batch_size 64 \
            --train_file "${train_file}" \
            --eval_file "${eval_file}" \
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
fi

# -------------------------------------------------------
# EVAL: evaluate.py (constrained beam search) + calc.py (HR/NDCG)
# Runs after SFT so each checkpoint is scored immediately.
# -------------------------------------------------------
if [[ "${PHASES}" == *"eval"* ]]; then
echo "===== EVAL PHASE ====="
mkdir -p results
SKIP_MODES="${SKIP_MODES:-}"
for mode in text fixed adaptive; do
    if [[ " ${SKIP_MODES} " == *" ${mode} "* ]]; then
        echo "--- EVAL: skipping mode=${mode} (in SKIP_MODES) ---"
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        echo "--- EVAL SFT: mode=${mode} seed=${seed} ---"
        # final_checkpoint holds model + tokenizer; top-level safetensors may
        # have been cleaned up to save disk
        ckpt="output_dir/sft_${mode}_seed${seed}/final_checkpoint"
        test_file=$(ls -f ${DATA_DIR[$mode]}/test/${DATASET}*11.csv)
        info_file=$(ls -f ${DATA_DIR[$mode]}/info/${DATASET}*.txt)
        result_json="results/eval_sft_${mode}_seed${seed}.json"

        python evaluate.py \
            --base_model "${ckpt}" \
            --info_file "${info_file}" \
            --category ${DATASET} \
            --test_data_path "${test_file}" \
            --result_json_data "${result_json}" \
            --num_beams 50 \
            --max_new_tokens 256 \
            --length_penalty 0.0 \
            --batch_size 8 \
            --seed ${seed}

        echo "--- CALC: mode=${mode} seed=${seed} ---"
        python calc.py \
            --path "${result_json}" \
            --item_path "${info_file}"
    done
done
fi

# -------------------------------------------------------
# GRPO: 2 methods (text, adaptive) x N seeds
# -------------------------------------------------------
if [[ "${PHASES}" == *"grpo"* ]]; then
echo "===== GRPO PHASE ====="
for mode in text adaptive; do
    for seed in "${SEEDS[@]}"; do
        echo "--- GRPO: mode=${mode} seed=${seed} ---"
        sft_ckpt="output_dir/sft_${mode}_seed${seed}/final_checkpoint"
        train_file=$(ls -f ${DATA_DIR[$mode]}/train/${DATASET}*.csv)
        eval_file=$(ls -f ${DATA_DIR[$mode]}/valid/${DATASET}*11.csv)
        info_file=$(ls -f ${DATA_DIR[$mode]}/info/${DATASET}*.txt)

        python ../acsid_amd/rl.py \
            --model_path "${sft_ckpt}" \
            --train_batch_size 64 \
            --eval_batch_size 128 \
            --num_train_epochs 2 \
            --gradient_accumulation_steps 2 \
            --train_file "${train_file}" \
            --eval_file "${eval_file}" \
            --info_file "${info_file}" \
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

        # free the middle checkpoint (~30GB each, has fp32 adamw_torch states)
        # before the next mode starts; keep final_checkpoint/ for eval
        rm -rf output_dir/grpo_${mode}_seed${seed}/checkpoint-*
    done
done
fi

echo "===== ALL DONE ====="
