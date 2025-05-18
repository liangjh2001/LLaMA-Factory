#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
# 默认模型路径
EMOTION_MODEL_PATH="/data/liangjh/LLaMA-Factory/output/Qwen2-Audio-7B-Instruct-audio_deepfake_train_noeval/checkpoint-1600/full-model"
DEEPFAKE_MODEL_PATH="/data/liangjh/LLaMA-Factory/output/Qwen2-Audio-7B-Instruct-audio_deepfake_train_noeval/checkpoint-1600/full-model"
SPEAKER_MODEL_PATH="/data/liangjh/LLaMA-Factory/output/Qwen2-Audio-7B-Instruct-audio_deepfake_train_noeval/checkpoint-1600/full-model"

# # 执行情绪识别任务
#echo "=================================================="
#echo "开始执行情绪识别任务"
#echo "=================================================="
#python qwen_audio_infer.py --task emotion --model_path "${EMOTION_MODEL_PATH}"

## wait
# 执行深度伪造检测任务
#echo ""
#echo "=================================================="
#echo "开始执行深度伪造检测任务"
#echo "=================================================="
#python qwen_audio_infer.py --task deepfake --model_path "${DEEPFAKE_MODEL_PATH}"

#wait
echo ""
echo "=================================================="
echo "开始执行说话人识别任务"
echo "=================================================="
python qwen_audio_infer.py --task speaker_recognition --model_path "${SPEAKER_MODEL_PATH}"

echo ""
echo "=================================================="
echo "所有任务已完成"
echo "==================================================" 