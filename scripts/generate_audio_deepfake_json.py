import json
import os
import random

# 定义路径
fake_dir = '/data/liangjh/LLaMA-Factory/data/FuseLLM/VFD/fake'
real_dir = '/data/liangjh/LLaMA-Factory/data/FuseLLM/VFD/real'
output_path = '/data/liangjh/LLaMA-Factory/data/audio_deepfake.json'
train_path = '/data/liangjh/LLaMA-Factory/data/audio_deepfake_train.json'
test_path = '/data/liangjh/LLaMA-Factory/data/audio_deepfake_test.json'

# 获取所有音频文件
def get_audio_files(directory, label_type):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(('.wav', '.mp3', '.flac')):
                # 直接使用指定格式的路径
                rel_path = f"FuseLLM/VFD/{label_type}/{filename}"
                files.append(rel_path)
    return files

# 获取伪造和真实音频文件
fake_files = get_audio_files(fake_dir, 'fake')
real_files = get_audio_files(real_dir, 'real')

print(f"找到伪造音频文件: {len(fake_files)}个")
print(f"找到真实音频文件: {len(real_files)}个")

# 打印一些样例路径以验证格式是否正确
if fake_files:
    print(f"伪造音频样例路径: {fake_files[0]}")
if real_files:
    print(f"真实音频样例路径: {real_files[0]}")

# 确保数据集平衡
# 如果一方数量较多，随机选取与另一方相同数量的样本
if len(fake_files) > len(real_files):
    fake_files = random.sample(fake_files, len(real_files))
elif len(real_files) > len(fake_files):
    real_files = random.sample(real_files, len(fake_files))

print(f"平衡后的伪造音频文件: {len(fake_files)}个")
print(f"平衡后的真实音频文件: {len(real_files)}个")

# 创建数据集
json_data = []

# 添加伪造音频数据
for audio_file in fake_files:
    conversation = {
        "messages": [
            {
                "content": "<audio>判断上述音频对话是否是AI生成的。请从以下选项中选择：是，否。",
                "role": "user"
            },
            {
                "content": "是",
                "role": "assistant"
            }
        ],
        "audios": [
            audio_file
        ]
    }
    json_data.append(conversation)

# 添加真实音频数据
for audio_file in real_files:
    conversation = {
        "messages": [
            {
                "content": "<audio>判断上述音频对话是否是AI生成的。请从以下选项中选择：是，否。",
                "role": "user"
            },
            {
                "content": "否",
                "role": "assistant"
            }
        ],
        "audios": [
            audio_file
        ]
    }
    json_data.append(conversation)

# 打乱数据
random.shuffle(json_data)

# 保存完整数据集
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"已生成JSON文件: {output_path}")
print(f"共处理了 {len(json_data)} 条音频数据")

# 分割训练集和测试集，保持两类数据的分布一致
fake_data = [item for item in json_data if item["messages"][1]["content"] == "是"]
real_data = [item for item in json_data if item["messages"][1]["content"] == "否"]

# 划分伪造数据
fake_train = fake_data[:int(len(fake_data) * 0.8)]
fake_test = fake_data[int(len(fake_data) * 0.8):]

# 划分真实数据
real_train = real_data[:int(len(real_data) * 0.8)]
real_test = real_data[int(len(real_data) * 0.8):]

# 组合并打乱训练集和测试集
train_data = fake_train + real_train
test_data = fake_test + real_test
random.shuffle(train_data)
random.shuffle(test_data)

# 保存训练集和测试集
with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f"已生成训练集: {train_path}，包含 {len(train_data)} 条数据")
print(f"其中伪造音频 {len(fake_train)} 条，真实音频 {len(real_train)} 条")
print(f"已生成测试集: {test_path}，包含 {len(test_data)} 条数据")
print(f"其中伪造音频 {len(fake_test)} 条，真实音频 {len(real_test)} 条") 