#!/bin/bash

output_dir="Llama-2-7b-hf-sft-wi_locness-full"
if [ ! -d "./output/${output_dir}" ]; then
  mkdir -p "./output/${output_dir}"
fi

cp $0 "./output/${output_dir}"

deepspeed --master_port=1222 --include localhost:3,4 src/train_bash.py \
    --deepspeed /data/liangjh/LLaMA-Factory-main/examples/deepspeed/ds_z3_offload_config.json \
    --stage sft \
    --model_name_or_path "/data/liangjh/model_set/Llama-2-7b-hf" \
    --do_train \
    --dataset wi_locness \
    --val_dataset bea_dev \
    --preprocessing_num_workers 60 \
    --cutoff_len 200 \
    --template gec \
    --finetuning_type full \
    --output_dir "./output/${output_dir}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy="epoch" \
    --save_strategy "epoch" \
    --save_only_model True \
    --num_train_epochs 5 \
    --report_to "tensorboard" \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --plot_loss \
    --bf16 \
    --log_level info \
    2>&1 | tee "./output/${output_dir}/${output_dir}.log" &