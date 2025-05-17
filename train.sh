#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
lr=5e-5
rank=8

# Task 2: Train speaker recognition dataset only with Qwen2-Audio-7B-Instruct
output_dir="Qwen2.5-Omni-7B-audio_speaker_recognition_train"
if [ ! -d "./output/${output_dir}" ]; then
  mkdir -p "./output/${output_dir}"
fi

cp $0 "./output/${output_dir}"

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port=32241 src/train.py \
    --stage sft \
    --model_name_or_path "/data/liangjh/model_set/Qwen2.5-Omni-7B" \
    --do_train \
    --dataset audio_speaker_recognition_train \
    --eval_dataset audio_speaker_recognition_test \
    --freeze_vision_tower \
    --preprocessing_num_workers 60 \
    --cutoff_len 3072 \
    --template qwen2_omni \
    --finetuning_type lora \
    --lora_rank ${rank} \
    --lora_target all \
    --output_dir "./output/${output_dir}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy="epoch" \
    --save_strategy="epoch" \
    --save_only_model True \
    --num_train_epochs 10 \
    --report_to "tensorboard" \
    --learning_rate ${lr} \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --plot_loss \
    --fp16 \
    --log_level info \
    2>&1 | tee "./output/${output_dir}/${output_dir}.log" &

wait



# Task 1: Train all three datasets (deepfake, emotion, speaker recognition) with Qwen2-Audio-7B-Instruct
output_dir="Qwen2.5-Omni-7B-audio_deepfake_emotion_speaker_train"
if [ ! -d "./output/${output_dir}" ]; then
  mkdir -p "./output/${output_dir}"
fi

cp $0 "./output/${output_dir}"

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port=32320 src/train.py \
    --stage sft \
    --model_name_or_path "/data/liangjh/model_set/Qwen2.5-Omni-7B" \
    --do_train \
    --dataset audio_deepfake_train,audio_emotion_train,audio_speaker_recognition_train \
    --eval_dataset audio_deepfake_test,audio_emotion_test,audio_speaker_recognition_test \
    --freeze_vision_tower \
    --preprocessing_num_workers 60 \
    --cutoff_len 3072 \
    --template qwen2_omni \
    --finetuning_type lora \
    --lora_rank ${rank} \
    --lora_target all \
    --output_dir "./output/${output_dir}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy="epoch" \
    --save_strategy="epoch" \
    --save_only_model True \
    --num_train_epochs 10 \
    --report_to "tensorboard" \
    --learning_rate ${lr} \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --plot_loss \
    --fp16 \
    --log_level info \
    2>&1 | tee "./output/${output_dir}/${output_dir}.log" &

wait


