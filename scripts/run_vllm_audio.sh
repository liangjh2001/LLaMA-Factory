#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# 默认模型路径
MODEL_PATH="/data/liangjh/LLaMA-Factory/output/Qwen2-Audio-7B-Instruct-audio_speaker_recognition_train/checkpoint-9450/full-model"


 # 执行情绪识别任务
#echo "=================================================="
#echo "开始执行情绪识别任务"
#echo "=================================================="
#python vllm_audio.py --task emotion --model_path "${MODEL_PATH}"

## wait
# 执行深度伪造检测任务
#echo ""
#echo "=================================================="
#echo "开始执行深度伪造检测任务"
#echo "=================================================="
#python vllm_audio.py --task deepfake --model_path "${MODEL_PATH}"
#
#wait
echo ""
echo "=================================================="
echo "开始执行说话人识别任务"
echo "=================================================="
python vllm_audio.py --task speaker_recognition --model_path "${MODEL_PATH}"

echo ""
echo "=================================================="
echo "所有任务已完成"
echo "=================================================="