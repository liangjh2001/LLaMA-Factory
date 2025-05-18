#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

base_model="/data/liangjh/model_set/Qwen2-Audio-7B-Instruct"
lora_model="Qwen2-Audio-7B-Instruct-audio_speaker_recognition_train"
template="qwen2_audio"
checkpoint="9450"


echo "----------------------------merge lora weight----------------------------"
python ./src/export_model.py \
    --model_name_or_path ${base_model} \
    --adapter_name_or_path "/data/liangjh/LLaMA-Factory/output/${lora_model}/checkpoint-${checkpoint}" \
    --template ${template} \
    --finetuning_type lora \
    --export_dir "/data/liangjh/LLaMA-Factory/output/${lora_model}/checkpoint-${checkpoint}/full-model" \
    --export_size 5 \
    --export_legacy_format False