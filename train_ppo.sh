#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export TRANSFORMERS_VERBOSITY="error"

lr=3e-5
rank=32

output_dir="Qwen2.5-0.5B-Instruct-ppo"
if [ ! -d "./output/${output_dir}" ]; then
  mkdir -p "./output/${output_dir}"
fi

cp $0 "./output/${output_dir}"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port=12220 src/train.py \
    --stage ppo \
    --model_name_or_path "/data/liangjh/model_set/Qwen2.5-0.5B-Instruct" \
    --reward_model "/data/liangjh/LLaMA-Factory/output/Qwen2.5-0.5B-Instruct-rm/checkpoint-170" \
    --do_train \
    --dataset alpaca_en_demo \
    --preprocessing_num_workers 60 \
    --cutoff_len 2048 \
    --max_samples 1000 \
    --template qwen \
    --finetuning_type lora \
    --lora_rank ${rank} \
    --lora_target all \
    --output_dir "./output/${output_dir}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy="steps" \
    --save_steps 100 \
    --save_only_model True \
    --num_train_epochs 1 \
    --report_to "tensorboard" \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --plot_loss \
    --bf16 \
    --log_level info \
    2>&1 | tee "./output/${output_dir}/${output_dir}.log" &

wait