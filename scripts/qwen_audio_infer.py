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
parser.add_argument('--task', type=str, default='emotion', choices=['deepfake', 'emotion', 'speaker_recognition'],
                   help='评估任务类型: deepfake (音频深度伪造检测) 或 emotion (情绪识别) 或 speaker_recognition (说话人识别)')
parser.add_argument('--model_path', type=str, 
                   default="/data/liangjh/LLaMA-Factory/output/Qwen2-Audio-7B-Instruct-audio_emotion_train/checkpoint-800/full-model",
                   help='预训练模型路径')
args = parser.parse_args()

# 加载预训练模型
model_load_path = args.model_path
processor = AutoProcessor.from_pretrained(model_load_path)
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_load_path, device_map="auto")

# --- Derive paths for output files based on model_load_path ---
checkpoint_dir_full = os.path.dirname(model_load_path) # e.g., .../checkpoint-800
base_output_dir = os.path.dirname(checkpoint_dir_full) # e.g., .../Qwen2-Audio-7B-Instruct-audio_emotion_train

# Define the target directory for results as a subdirectory named 'results'
output_file_target_dir = os.path.join(base_output_dir, "results")

model_name_for_filename = os.path.basename(base_output_dir) # e.g., Qwen2-Audio-7B-Instruct-audio_emotion_train
checkpoint_folder_name = os.path.basename(checkpoint_dir_full) # e.g., checkpoint-800
checkpoint_id_for_filename = checkpoint_folder_name.replace("checkpoint-", "ckpt-") # e.g., ckpt-800

# Ensure the target directory for output files (including the 'results' subdirectory) exists
os.makedirs(output_file_target_dir, exist_ok=True)
# --- End of path derivation ---

# 根据任务类型设置路径
if args.task == 'deepfake':
    test_path = '/data/liangjh/LLaMA-Factory/data/audio_deepfake_test.json'
    results_filename = f"audio_deepfake_results_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    results_path = os.path.join(output_file_target_dir, results_filename)
    report_filename = f"audio_deepfake_evaluation_report_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    report_path = os.path.join(output_file_target_dir, report_filename)
    prompt = "判断上述音频对话是否是AI生成的。请从以下选项中选择：是，否。"
    print(f"执行音频深度伪造检测任务")
elif args.task == 'emotion':
    test_path = '/data/liangjh/LLaMA-Factory/data/audio_emotion_test.json'
    results_filename = f"audio_emotion_results_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    results_path = os.path.join(output_file_target_dir, results_filename)
    report_filename = f"audio_emotion_evaluation_report_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    report_path = os.path.join(output_file_target_dir, report_filename)
    prompt = "识别上述音频对话中表达的情感。请从以下选项中选择：愤怒，开心，厌恶，中性。"
    print(f"执行音频情绪识别任务")
else:  # speaker_recognition
    test_path = '/data/liangjh/LLaMA-Factory/data/audio_speaker_recognition_test.json'
    results_filename = f"audio_speaker_recognition_results_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    results_path = os.path.join(output_file_target_dir, results_filename)
    report_filename = f"audio_speaker_recognition_evaluation_report_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    report_path = os.path.join(output_file_target_dir, report_filename)
    prompt = "判断上述两个音频对话是否来自同一说话人。请从以下选项中选择：是，否。"
    print(f"执行说话人识别任务")

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
else:  # deepfake or speaker_recognition
    positive_class_correct = 0
    positive_class_total = 0
    negative_class_correct = 0
    negative_class_total = 0

# 结果记录
results = []
true_labels = []
pred_labels = []

# 遍历测试数据并进行推理
for sample in tqdm(test_data, desc="处理音频样本"):
    try:
        true_label = sample["messages"][1]["content"]
        audios_for_processor = []
        
        if args.task == 'speaker_recognition':
            audio_rel_path_1 = sample["audios"][0]
            audio_full_path_1 = os.path.join(audio_base_dir, audio_rel_path_1)
            audio_rel_path_2 = sample["audios"][1]
            audio_full_path_2 = os.path.join(audio_base_dir, audio_rel_path_2)

            # 为说话人识别任务构建正确的conversation格式
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                {"role": "user", "content": [
                    {"type": "text", "text": "音频1"},
                    {"type": "audio", "audio_url": audio_full_path_1},
                    {"type": "text", "text": "音频2"},
                    {"type": "audio", "audio_url": audio_full_path_2},
                    {"type": "text", "text": prompt}, # 使用纯文本prompt
                ]},
            ]
            audio_array_1, _ = librosa.load(audio_full_path_1, sr=processor.feature_extractor.sampling_rate)
            audios_for_processor.append(audio_array_1)
            audio_array_2, _ = librosa.load(audio_full_path_2, sr=processor.feature_extractor.sampling_rate)
            audios_for_processor.append(audio_array_2)
            display_audio_rel_path = f"{audio_rel_path_1}, {audio_rel_path_2}"
        else: # deepfake or emotion (single audio tasks)
            audio_rel_path = sample["audios"][0]
            audio_full_path = os.path.join(audio_base_dir, audio_rel_path)
            display_audio_rel_path = audio_rel_path

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_full_path},
                    {"type": "text", "text": prompt},
                ]},
            ]
            audio_array, _ = librosa.load(audio_full_path, sr=processor.feature_extractor.sampling_rate)
            audios_for_processor.append(audio_array)

        # 准备输入
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        inputs = processor(text=text, audios=audios_for_processor, return_tensors="pt", padding=True)
        inputs.input_ids = inputs.input_ids.to("cuda")
        
        generate_ids = model.generate(**inputs, max_length=512)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        
        predicted_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        if args.task == 'deepfake' or args.task == 'speaker_recognition':
            predicted_label = "是" if "是" in predicted_text and "否" not in predicted_text else "否"
        else:  # emotion
            for label_option in labels: # Changed from `label` to `label_option` to avoid conflict
                if label_option in predicted_text:
                    predicted_label = label_option
                    break
            else:
                predicted_label = "中性"
        
        results.append({
            "audio_url": display_audio_rel_path, 
            "true_label": true_label,
            "predicted_label": predicted_label,
            "raw_prediction": predicted_text
        })
        
        true_labels.append(true_label)
        pred_labels.append(predicted_label)
        
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct_predictions += 1
        
        if args.task == 'deepfake' or args.task == 'speaker_recognition':
            if true_label == "是":
                positive_class_total += 1
                if is_correct:
                    positive_class_correct += 1
            else: 
                negative_class_total += 1
                if is_correct:
                    negative_class_correct += 1
        else:  # emotion
            class_total[true_label] = class_total.get(true_label, 0) + 1
            if is_correct:
                class_correct[true_label] = class_correct.get(true_label, 0) + 1
        
        total_predictions += 1
        
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

if args.task == 'deepfake' or args.task == 'speaker_recognition':
    positive_label_name = "AI生成" if args.task == 'deepfake' else "同一说话人"
    negative_label_name = "真实音频" if args.task == 'deepfake' else "不同说话人"

    positive_accuracy = positive_class_correct / positive_class_total if positive_class_total > 0 else 0
    negative_accuracy = negative_class_correct / negative_class_total if negative_class_total > 0 else 0
    
    print(f"{positive_label_name}的准确率: {positive_accuracy:.4f} ({positive_class_correct}/{positive_class_total})")
    print(f"{negative_label_name}的准确率: {negative_accuracy:.4f} ({negative_class_correct}/{negative_class_total})")
    
    true_positives = positive_class_correct
    false_positives = negative_class_total - negative_class_correct
    false_negatives = positive_class_total - positive_class_correct
    true_negatives = negative_class_correct
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    
    print("\n混淆矩阵:")
    print(f"真实\预测 | 是 ({positive_label_name}) | 否 ({negative_label_name})")
    print(f"是 ({positive_label_name}) | {true_positives:12d} | {false_negatives:14d}")
    print(f"否 ({negative_label_name})| {false_positives:12d} | {true_negatives:14d}")
    
    report.update({
        f"{positive_label_name}_样本数": positive_class_total,
        f"{positive_label_name}_正确预测数": positive_class_correct,
        f"{positive_label_name}_准确率": positive_accuracy,
        f"{negative_label_name}_样本数": negative_class_total,
        f"{negative_label_name}_正确预测数": negative_class_correct,
        f"{negative_label_name}_准确率": negative_accuracy,
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
    
elif args.task == 'emotion': # Corrected from else to elif
    emotion_accuracies = {}
    for label_option_report in labels: # Changed variable name to avoid conflict
        if class_total.get(label_option_report, 0) > 0:
            acc = class_correct.get(label_option_report, 0) / class_total.get(label_option_report, 0)
            emotion_accuracies[label_option_report] = acc
            print(f"{label_option_report}的准确率: {acc:.4f} ({class_correct.get(label_option_report, 0)}/{class_total.get(label_option_report, 0)})")
    
    class_report_dict = classification_report(true_labels, pred_labels, labels=labels, output_dict=True, zero_division=0) # Added labels and zero_division
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels, labels=labels, zero_division=0)) # Added labels and zero_division
    
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    print("\n混淆矩阵:")
    print("预测 →")
    print("真实 ↓  " + " ".join([f"{label_cm:8s}" for label_cm in labels])) # Changed variable name
    for i, label_row in enumerate(labels): # Changed variable name
        print(f"{label_row:6s} " + " ".join([f"{cm[i, j]:8d}" for j in range(len(labels))]))
    
    report.update({
        "各情绪类别准确率": emotion_accuracies,
        "分类报告": class_report_dict, # use the dict version
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

