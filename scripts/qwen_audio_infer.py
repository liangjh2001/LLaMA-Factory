from io import BytesIO
import os
import json
import librosa
import numpy as np
import argparse
from tqdm import tqdm
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from sklearn.metrics import confusion_matrix, classification_report

# 解析命令行参数
parser = argparse.ArgumentParser(description='Qwen音频推理评估')
parser.add_argument('--task', type=str, default='emotion', choices=['deepfake', 'emotion'],
                   help='评估任务类型: deepfake (音频深度伪造检测) 或 emotion (情绪识别)')
parser.add_argument('--model_path', type=str, 
                   default="/data/liangjh/LLaMA-Factory/output/Qwen2-Audio-7B-Instruct-audio_emotion_train/checkpoint-800/full-model",
                   help='预训练模型路径')
args = parser.parse_args()

# 加载预训练模型
path = args.model_path
#path = "/data/liangjh/LLaMA-Factory/output/Qwen2-Audio-7B-Instruct-audio_deepfake_train/checkpoint-1600/full-model"
processor = AutoProcessor.from_pretrained(path)
model = Qwen2AudioForConditionalGeneration.from_pretrained(path, device_map="auto")

# 根据任务类型设置路径
if args.task == 'deepfake':
    test_path = '/data/liangjh/LLaMA-Factory/data/audio_deepfake_test.json'
    results_path = '/data/liangjh/LLaMA-Factory/data/audio_deepfake_results.json'
    report_path = '/data/liangjh/LLaMA-Factory/data/audio_deepfake_evaluation_report.json'
    prompt = "判断上述音频对话是否是AI生成的。请从以下选项中选择：是，否。"
    print(f"执行音频深度伪造检测任务")
else:  # emotion
    test_path = '/data/liangjh/LLaMA-Factory/data/audio_emotion_test.json'
    results_path = '/data/liangjh/LLaMA-Factory/data/audio_emotion_results.json'
    report_path = '/data/liangjh/LLaMA-Factory/data/audio_emotion_evaluation_report.json'
    prompt = "识别上述音频对话中表达的情感。请从以下选项中选择：愤怒，开心，厌恶，中性。"
    print(f"执行音频情绪识别任务")

audio_base_dir = '/data/liangjh/LLaMA-Factory/data/'  # 音频文件的基础目录

# 读取测试集数据
with open(test_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# test_data = test_data[:10]  # 限制测试数据集大小为100条
print(f"加载了 {len(test_data)} 条测试数据")

# 初始化统计信息
correct_predictions = 0
total_predictions = 0

# 如果是情绪识别任务，需要跟踪每个类别
if args.task == 'emotion':
    labels = ['愤怒', '开心', '厌恶', '中性']
    class_correct = {label: 0 for label in labels}
    class_total = {label: 0 for label in labels}
else:  # deepfake
    fake_correct = 0
    fake_total = 0
    real_correct = 0
    real_total = 0

# 结果记录
results = []
true_labels = []
pred_labels = []

# 遍历测试数据并进行推理
for sample in tqdm(test_data, desc="处理音频样本"):
    try:
        # 获取音频文件路径和真实标签
        audio_rel_path = sample["audios"][0]
        audio_full_path = os.path.join(audio_base_dir, audio_rel_path)
        true_label = sample["messages"][1]["content"]
        
        # 创建会话格式
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_full_path},
                {"type": "text", "text": prompt},
            ]},
        ]
        
        # 准备输入
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        
        # 加载音频文件
        audio_array, _ = librosa.load(audio_full_path, sr=processor.feature_extractor.sampling_rate)
        audios.append(audio_array)
        
        # 处理输入并生成
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs.input_ids = inputs.input_ids.to("cuda")
        
        generate_ids = model.generate(**inputs, max_length=512)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        
        # 解码预测结果
        predicted_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # 根据任务类型处理预测结果
        if args.task == 'deepfake':
            # 简单处理预测结果，确保其中只包含"是"或"否"
            predicted_label = "是" if "是" in predicted_text and "否" not in predicted_text else "否"
        else:  # emotion
            # 查找情绪标签
            for label in labels:
                if label in predicted_text:
                    predicted_label = label
                    break
            else:
                # 如果没有找到匹配的标签，选择一个默认值
                predicted_label = "中性"
        
        # 记录结果
        results.append({
            "audio_url": audio_rel_path,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "raw_prediction": predicted_text
        })
        
        # 保存用于生成混淆矩阵的标签
        true_labels.append(true_label)
        pred_labels.append(predicted_label)
        
        # 检查预测是否正确
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct_predictions += 1
        
        # 根据任务类型更新各类别的统计数据
        if args.task == 'deepfake':
            if true_label == "是":  # AI生成的音频
                fake_total += 1
                if is_correct:
                    fake_correct += 1
            else:  # 真实音频
                real_total += 1
                if is_correct:
                    real_correct += 1
        else:  # emotion
            class_total[true_label] = class_total.get(true_label, 0) + 1
            if is_correct:
                class_correct[true_label] = class_correct.get(true_label, 0) + 1
        
        total_predictions += 1
        
        # 打印进度
        if total_predictions % 10 == 0:
            print(f"当前准确率: {correct_predictions/total_predictions:.4f}")
            
    except Exception as e:
        print(f"处理样本时出错: {e}")
        continue

# 计算总体准确率
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"\n总体准确率: {accuracy:.4f} ({correct_predictions}/{total_predictions})")

# 根据任务类型生成报告
report = {
    "总样本数": total_predictions,
    "总体准确率": accuracy,
}

if args.task == 'deepfake':
    # 计算各类别的准确率
    fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
    real_accuracy = real_correct / real_total if real_total > 0 else 0
    
    print(f"AI生成音频的准确率: {fake_accuracy:.4f} ({fake_correct}/{fake_total})")
    print(f"真实音频的准确率: {real_accuracy:.4f} ({real_correct}/{real_total})")
    
    # 计算更多评估指标
    # 将"是"（AI生成）视为正类
    true_positives = fake_correct  # 正确预测为"是"的样本数
    false_positives = real_total - real_correct  # 错误预测为"是"的样本数
    false_negatives = fake_total - fake_correct  # 错误预测为"否"的样本数
    true_negatives = real_correct  # 正确预测为"否"的样本数
    
    # 精确率 = TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # 召回率 = TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    print(f"真实\\预测 | 是（AI生成） | 否（真实音频）")
    print(f"是（AI生成） | {true_positives:12d} | {false_negatives:14d}")
    print(f"否（真实音频）| {false_positives:12d} | {true_negatives:14d}")
    
    # 添加到报告
    report.update({
        "AI生成音频样本数": fake_total,
        "AI生成音频正确预测数": fake_correct,
        "AI生成音频准确率": fake_accuracy,
        "真实音频样本数": real_total,
        "真实音频正确预测数": real_correct,
        "真实音频准确率": real_accuracy,
        "精确率": precision,
        "召回率": recall,
        "F1分数": f1_score,
        "混淆矩阵": {
            "真正例(TP)": int(true_positives),
            "假正例(FP)": int(false_positives),
            "假负例(FN)": int(false_negatives),
            "真负例(TN)": int(true_negatives)
        }
    })
    
else:  # emotion
    # 计算每个情绪类别的准确率
    emotion_accuracies = {}
    for label in labels:
        if class_total.get(label, 0) > 0:
            accuracy = class_correct.get(label, 0) / class_total.get(label, 0)
            emotion_accuracies[label] = accuracy
            print(f"{label}的准确率: {accuracy:.4f} ({class_correct.get(label, 0)}/{class_total.get(label, 0)})")
    
    # 使用sklearn生成详细的分类报告
    class_report = classification_report(true_labels, pred_labels, output_dict=True)
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels))
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    print("\n混淆矩阵:")
    print("预测 →")
    print("真实 ↓  " + " ".join([f"{label:8s}" for label in labels]))
    for i, label in enumerate(labels):
        print(f"{label:6s} " + " ".join([f"{cm[i, j]:8d}" for j in range(len(labels))]))
    
    # 添加到报告
    report.update({
        "各情绪类别准确率": emotion_accuracies,
        "分类报告": class_report,
        "混淆矩阵": cm.tolist()
    })

# 保存详细结果
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"详细结果已保存到: {results_path}")

# 保存评估报告
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"评估报告已保存到: {report_path}")
