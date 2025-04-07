#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

for lr in 3e-5; do


output_dir="Qwen2.5-0.5B-Instruct-dpo-new-sorted_loss"
if [ ! -d "./output/${output_dir}" ]; then
  mkdir -p "./output/${output_dir}"
fi

cp $0 "./output/${output_dir}"


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port=29299 src/train.py \
    --deepspeed /data/liangjh/LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
    --stage dpo \
    --pref_loss sigmoid \
    --pref_beta 0.1 \
    --pref_ftx 0.1 \
    --model_name_or_path "/data/liangjh/LLaMA-Factory/output/Qwen2.5-0.5B-Instruct-test/checkpoint-16/full-model" \
    --do_train \
    --dataset dpo_en_demo \
    --val_size 0.1 \
    --preprocessing_num_workers 60 \
    --cutoff_len 200 \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 32 \
    --lora_target all \
    --output_dir "./output/${output_dir}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy="steps" \
    --eval_steps 0.5 \
    --save_strategy="steps" \
    --save_steps 0.5 \
    --save_only_model True \
    --num_train_epochs 10 \
    --warmup_ratio 0.1 \
    --report_to "tensorboard" \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --plot_loss \
    --bf16 \
    --log_level info \
    2>&1 | tee "./output/${output_dir}/${output_dir}.log" &

wait
done
