import pandas as pd
import json
import os
import random

# 读取Excel文件
excel_path = '/data/liangjh/LLaMA-Factory/data/FuseLLM/emotion/1000_audio-ASR-emo.xlsx'
df = pd.read_excel(excel_path)

# 创建用于JSON的数据结构
json_data = []

# 遍历Excel文件中的每一行
for _, row in df.iterrows():
    wav_name = row['wav_name']
    label = row['label']
    wav_name = 'FuseLLM/emotion/1000_audio/' + wav_name
    # 创建对话数据
    conversation = {
        "messages": [
            {
                "content": "<audio>识别上述音频对话中表达的情感。请从以下选项中选择：愤怒，开心，厌恶，中性。",
                "role": "user"
            },
            {
                "content": label,
                "role": "assistant"
            }
        ],
        "audios": [
            wav_name
        ]
    }
    
    json_data.append(conversation)

# 保存为JSON文件
output_path = '/data/liangjh/LLaMA-Factory/data/audio_emotion.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"已生成JSON文件: {output_path}")
print(f"共处理了 {len(json_data)} 条音频情感标注数据") 

# 按8：2的比例划分训练集和测试集
# 随机打乱数据
random.shuffle(json_data)

# 按8：2的比例划分训练集和测试集
train_data = json_data[:int(len(json_data) * 0.8)]
test_data = json_data[int(len(json_data) * 0.8):]

# 保存训练集和测试集
train_path = '/data/liangjh/LLaMA-Factory/data/audio_emotion_train.json'
test_path = '/data/liangjh/LLaMA-Factory/data/audio_emotion_test.json'

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
