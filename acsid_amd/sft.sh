#!/bin/bash
# ACSID SFT — AMD MI300X 192GB single GPU
# No torchrun, no DeepSpeed needed. 192GB VRAM is enough for full-param 3B.
# 16 (micro) x 64 (accum) = 1024 effective batch (same as original 8-card config)

export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES=0

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}

    python sft.py \
            --base_model /path/to/Qwen2.5-3B-Base \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
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