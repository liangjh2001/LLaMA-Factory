import json
import os
import random

# 定义路径
# 音频文件现在位于切分后的目录
voice_dir_actual = '/data/liangjh/LLaMA-Factory/data/FuseLLM/voice_split/' 
json_audio_prefix = 'FuseLLM/voice_split/' # JSON中引用的音频路径前缀

output_path = '/data/liangjh/LLaMA-Factory/data/audio_speaker_recognition.json'
train_path = '/data/liangjh/LLaMA-Factory/data/audio_speaker_recognition_train.json'
test_path = '/data/liangjh/LLaMA-Factory/data/audio_speaker_recognition_test.json'

AUDIO_SUFFIX_A = "_A.amr"
AUDIO_SUFFIX_B = "_B.amr"

def get_audio_basenames_from_split_dir(directory, suffix_a, suffix_b):
    """
    从已切分的音频目录中获取音频基本名。
    它会查找 suffix_a 文件，并验证相应的 suffix_b 文件是否存在。
    """
    basenames = set()
    if not os.path.exists(directory):
        print(f"错误：目录 {directory} 不存在。请确保路径正确且音频已切分到此目录。")
        return list(basenames)

    files_in_dir = os.listdir(directory)
    found_a_parts_count = 0
    
    for filename in files_in_dir:
        if filename.endswith(suffix_a):
            found_a_parts_count += 1
            base = filename[:-len(suffix_a)] # 例如从 'file1_A.amr' 提取 'file1'
            # 检查对应的 _B 部分是否存在
            if f"{base}{suffix_b}" in files_in_dir:
                basenames.add(base)
            else:
                print(f"警告：找到 {filename} 但未找到对应的 {base}{suffix_b}。将忽略此文件对 '{base}'。")
    
    if found_a_parts_count == 0:
        print(f"警告：在目录 {directory} 中没有找到任何以 '{suffix_a}' 结尾的音频文件。")
        print(f"请确保您已运行切分脚本并将 '{suffix_a}' 和 '{suffix_b}' 文件放入该目录。")
    elif not basenames:
        print(f"警告：在目录 {directory} 中找到了 {found_a_parts_count} 个 '{suffix_a}' 文件，但没有形成有效的 '{suffix_a}'/'{suffix_b}' 对。")
        print(f"请检查文件命名和完整性。")
            
    return list(basenames)

# 获取所有原始音频文件的基本名 (从切分后的目录中获取)
original_audio_basenames = get_audio_basenames_from_split_dir(voice_dir_actual, AUDIO_SUFFIX_A, AUDIO_SUFFIX_B)

if not original_audio_basenames:
    print("未能从切分目录获取任何有效的音频文件基本名，脚本将退出。")
    print(f"请检查目录 '{voice_dir_actual}' 是否包含成对的 '{AUDIO_SUFFIX_A}' 和 '{AUDIO_SUFFIX_B}' 文件。")
    exit()

print(f"从 '{voice_dir_actual}' 找到 {len(original_audio_basenames)} 个有效的音频基本名 (例如，基于 *{AUDIO_SUFFIX_A}/*{AUDIO_SUFFIX_B} 对)。")
if original_audio_basenames:
    print(f"部分样例基本名: {original_audio_basenames[:5]}")

# 创建数据集
json_data = []

# 构建数据
for i, base_name in enumerate(original_audio_basenames):
    part_a_filename = f"{base_name}{AUDIO_SUFFIX_A}"
    part_b_filename = f"{base_name}{AUDIO_SUFFIX_B}"

    path_a = f"{json_audio_prefix}{part_a_filename}"
    path_b = f"{json_audio_prefix}{part_b_filename}"

    # 1. 创建正样本 (同一说话人的两个不同片段)
    positive_conversation = {
        "messages": [
            {
                "content": "音频1<audio>音频2<audio>判断上述两个音频对话是否来自同一说话人。请从以下选项中选择：是，否。",
                "role": "user"
            },
            {
                "content": "是",
                "role": "assistant"
            }
        ],
        "audios": [path_a, path_b]
    }
    json_data.append(positive_conversation)

    # 2. 创建负样本 (当前说话人的片段A vs 其他说话人的片段A)
    other_basenames = [b for b in original_audio_basenames if b != base_name]
    
    num_negative_samples = 3
    if len(other_basenames) < num_negative_samples:
        # print(f"信息: 说话人 {base_name} 的可用其他说话人不足 {num_negative_samples} 个（实际可用 {len(other_basenames)} 个）。将使用所有可用的。")
        selected_negative_basenames = other_basenames
    else:
        selected_negative_basenames = random.sample(other_basenames, num_negative_samples)

    for neg_base_name in selected_negative_basenames:
        neg_part_a_filename = f"{neg_base_name}{AUDIO_SUFFIX_A}"
        neg_path_a = f"{json_audio_prefix}{neg_part_a_filename}"
        
        negative_conversation = {
            "messages": [
                {
                    "content": "音频1<audio>音频2<audio>判断上述两个音频对话是否来自同一说话人。请从以下选项中选择：是，否。",
                    "role": "user"
                },
                {
                    "content": "否",
                    "role": "assistant"
                }
            ],
            "audios": [path_a, neg_path_a] # 当前说话人的 part_a 和 其他说话人的 part_a
        }
        json_data.append(negative_conversation)
    
    if (i + 1) % 100 == 0:
        print(f"已处理 {i + 1}/{len(original_audio_basenames)} 个基本音频名 (生成了正负样本)... ")

# 打乱数据
random.shuffle(json_data)

# 保存完整数据集
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"已生成完整JSON文件: {output_path}")
print(f"共生成了 {len(json_data)} 条对话数据。")

# --- 修改开始：采用分层抽样，确保测试集200条且分布一致 ---

# 1. 分离正负样本
all_positive_samples = [item for item in json_data if item["messages"][1]["content"] == "是"]
all_negative_samples = [item for item in json_data if item["messages"][1]["content"] == "否"]

# 2. 分别打乱
random.shuffle(all_positive_samples)
random.shuffle(all_negative_samples)

# 3. 定义测试集目标数量
num_test_samples_total_target = 200
num_positive_total = len(all_positive_samples)
num_negative_total = len(all_negative_samples)

if num_positive_total + num_negative_total == 0:
    print("错误：没有可供划分的数据！")
    exit()

# 计算测试集中各类别的目标数量，保持整体1:3的比例
# (num_positive_total / (num_positive_total + num_negative_total)) 约等于 1/4
# (num_negative_total / (num_positive_total + num_negative_total)) 约等于 3/4

# 确保至少有足够的样本进行分配
if num_test_samples_total_target > (num_positive_total + num_negative_total):
    print(f"警告：期望的测试集数量 ({num_test_samples_total_target}) 大于总样本数 ({(num_positive_total + num_negative_total)})。将使用所有数据作为训练集，测试集为空。")
    test_positive_count = 0
    test_negative_count = 0
elif num_positive_total == 0 or num_negative_total == 0: # 处理只有一类样本的极端情况
    print("警告：数据集中只存在一类样本，无法按比例分配到测试集。将按总数8:2划分。")
    # Fallback to simple 80/20 split if only one class exists for simplicity
    # This case should ideally not happen with the current data generation logic (1 pos, 3 neg)
    simple_test_count = int(len(json_data) * 0.2)
    test_data = json_data[:simple_test_count]
    train_data = json_data[simple_test_count:]
    # Re-calculate counts for print statements later
    test_positive_count = sum(1 for item in test_data if item["messages"][1]["content"] == "是")
    test_negative_count = len(test_data) - test_positive_count
else:
    # 正常按比例分配
    test_positive_count = round(num_test_samples_total_target * (num_positive_total / (num_positive_total + num_negative_total)))
    test_negative_count = num_test_samples_total_target - test_positive_count # 确保总数是200

    # 再次检查，确保不会取超出范围的样本
    test_positive_count = min(test_positive_count, num_positive_total)
    test_negative_count = min(test_negative_count, num_negative_total)
    
    # 如果因为取整或min导致总数不足200，尝试调整以接近200，优先保证比例
    current_test_total = test_positive_count + test_negative_count
    if current_test_total < num_test_samples_total_target and current_test_total > 0:
        remaining_needed = num_test_samples_total_target - current_test_total
        # 尝试按比例分配剩余部分
        if num_positive_total - test_positive_count > 0 and num_negative_total - test_negative_count > 0:
            add_pos = round(remaining_needed * ( (num_positive_total - test_positive_count) / ((num_positive_total - test_positive_count) + (num_negative_total - test_negative_count)) ))
            add_neg = remaining_needed - add_pos
            test_positive_count += min(add_pos, num_positive_total - test_positive_count)
            test_negative_count += min(add_neg, num_negative_total - test_negative_count)
        elif num_positive_total - test_positive_count > 0: # 只能加正样本
            test_positive_count += min(remaining_needed, num_positive_total - test_positive_count)
        elif num_negative_total - test_negative_count > 0: # 只能加负样本
            test_negative_count += min(remaining_needed, num_negative_total - test_negative_count)

# 4. 划分样本
if 'simple_test_count' not in locals(): # 如果不是上面极端情况的fallback
    test_positive_samples = all_positive_samples[:test_positive_count]
    train_positive_samples = all_positive_samples[test_positive_count:]

    test_negative_samples = all_negative_samples[:test_negative_count]
    train_negative_samples = all_negative_samples[test_negative_count:]

    # 5. 合并为训练集和测试集
    train_data = train_positive_samples + train_negative_samples
    test_data = test_positive_samples + test_negative_samples

    # 6. 分别打乱最终的训练集和测试集
    random.shuffle(train_data)
    random.shuffle(test_data)

# --- 修改结束 ---

# 保存训练集和测试集
os.makedirs(os.path.dirname(train_path), exist_ok=True)
os.makedirs(os.path.dirname(test_path), exist_ok=True)

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

train_pos_count = sum(1 for item in train_data if item["messages"][1]["content"] == "是")
train_neg_count = len(train_data) - train_pos_count
test_pos_count = sum(1 for item in test_data if item["messages"][1]["content"] == "是")
test_neg_count = len(test_data) - test_pos_count

print(f"已生成训练集: {train_path}，包含 {len(train_data)} 条数据 (正样本: {train_pos_count}, 负样本: {train_neg_count})")
print(f"已生成测试集: {test_path}，包含 {len(test_data)} 条数据 (正样本: {test_pos_count}, 负样本: {test_neg_count})")

print("\n重要提示:")
print(f"此脚本现在从 '{voice_dir_actual}' 读取已切分的 '{AUDIO_SUFFIX_A}' 和 '{AUDIO_SUFFIX_B}' 文件。")
print(f"请确保您已使用 'scripts/split_amr_audio_files.py' (或类似方法) 完成音频切分。") 