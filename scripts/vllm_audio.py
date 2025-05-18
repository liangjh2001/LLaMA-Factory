import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU device ID
os.environ['VLLM_USE_V1'] = '0'
import json
import librosa
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoProcessor
from sklearn.metrics import confusion_matrix, classification_report
import warnings
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 解析命令行参数
parser = argparse.ArgumentParser(description='VLLM Qwen音频推理评估')
parser.add_argument('--task', type=str, default='deepfake', choices=['deepfake', 'emotion', 'speaker_recognition'],
                   help='评估任务类型: deepfake (音频深度伪造检测) 或 emotion (情绪识别) 或 speaker_recognition (说话人识别)')
parser.add_argument('--model_path', type=str,
                   default="/data/liangjh/LLaMA-Factory/output/Qwen2-Audio-7B-Instruct-audio_deepfake_emotion_speaker_train/checkpoint-6180/full-model", # Example path, user will provide
                   help='预训练模型路径')
args = parser.parse_args()

# 加载 Processor
processor = AutoProcessor.from_pretrained(args.model_path)

# 配置vLLM引擎参数
# 根据任务确定最大音频输入数量
max_audio_inputs = 1
if args.task == 'speaker_recognition':
    max_audio_inputs = 2

engine_args = EngineArgs(
    model=args.model_path,
    max_model_len=4096,  # 与qwen_audio_infer保持一致或根据需要调整
    max_num_seqs=10,     # 可根据批处理大小调整
    limit_mm_per_prompt={"audio": max_audio_inputs},
    trust_remote_code=True # Often needed for custom models/processors
)
llm = LLM(
    model=args.model_path,
    max_model_len=4096,  # 与qwen_audio_infer保持一致或根据需要调整
    max_num_seqs=10,  # 可根据批处理大小调整
    limit_mm_per_prompt={"audio": max_audio_inputs},
    trust_remote_code=True  # Often needed for custom models/processors
) # Corrected: pass engine_args directly

# --- Derive paths for output files based on model_load_path ---
checkpoint_dir_full = os.path.dirname(args.model_path) # e.g., .../checkpoint-800/full-model -> .../checkpoint-800
if os.path.basename(args.model_path) == "full-model": # handle paths like .../checkpoint-xxxx/full-model
    checkpoint_dir_full = os.path.dirname(args.model_path)
else: # handle paths like .../checkpoint-xxxx
    checkpoint_dir_full = args.model_path

base_output_dir = os.path.dirname(checkpoint_dir_full) # e.g., .../Qwen2-Audio-7B-Instruct-audio_emotion_train

# Define the target directory for results as a subdirectory named 'results_vllm'
output_file_target_dir = os.path.join(base_output_dir, "results_vllm") # Added _vllm to distinguish

model_name_for_filename = os.path.basename(base_output_dir)
checkpoint_folder_name = os.path.basename(checkpoint_dir_full)
checkpoint_id_for_filename = checkpoint_folder_name.replace("checkpoint-", "ckpt-")

os.makedirs(output_file_target_dir, exist_ok=True)
# --- End of path derivation ---

# 根据任务类型设置路径和提示
if args.task == 'deepfake':
    test_path = '/data/liangjh/LLaMA-Factory/data/audio_deepfake_test.json'
    results_filename = f"audio_deepfake_results_vllm_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    results_path = os.path.join(output_file_target_dir, results_filename)
    report_filename = f"audio_deepfake_evaluation_report_vllm_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    report_path = os.path.join(output_file_target_dir, report_filename)
    base_prompt_text = "判断上述音频对话是否是AI生成的。请从以下选项中选择：是，否。"
    print(f"执行音频深度伪造检测任务 (vLLM)")
elif args.task == 'emotion':
    test_path = '/data/liangjh/LLaMA-Factory/data/audio_emotion_test.json'
    results_filename = f"audio_emotion_results_vllm_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    results_path = os.path.join(output_file_target_dir, results_filename)
    report_filename = f"audio_emotion_evaluation_report_vllm_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    report_path = os.path.join(output_file_target_dir, report_filename)
    base_prompt_text = "识别上述音频对话中表达的情感。请从以下选项中选择：愤怒，开心，厌恶，中性。"
    print(f"执行音频情绪识别任务 (vLLM)")
else:  # speaker_recognition
    test_path = '/data/liangjh/LLaMA-Factory/data/audio_speaker_recognition_test.json'
    results_filename = f"audio_speaker_recognition_results_vllm_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    results_path = os.path.join(output_file_target_dir, results_filename)
    report_filename = f"audio_speaker_recognition_evaluation_report_vllm_{model_name_for_filename}_{checkpoint_id_for_filename}.json"
    report_path = os.path.join(output_file_target_dir, report_filename)
    base_prompt_text = "判断上述两个音频对话是否来自同一说话人。请从以下选项中选择：是，否。"
    print(f"执行说话人识别任务 (vLLM)")

audio_base_dir = '/data/liangjh/LLaMA-Factory/data/'

# 读取测试集数据
with open(test_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# test_data = test_data[:5] # 限制测试数据集大小
print(f"加载了 {len(test_data)} 条测试数据")

# 初始化统计信息
correct_predictions = 0
total_predictions = 0
results = []
true_labels_list = [] # Renamed from true_labels to avoid conflict with variable in loop
pred_labels_list = [] # Renamed from pred_labels to avoid conflict

# 用于批量推理的列表
all_inputs_for_vllm = []
all_true_labels_for_batch = []
all_display_audio_paths_for_batch = []
all_original_samples_for_batch = [] # Store original sample for error reporting if needed

if args.task == 'emotion':
    emotion_labels = ['愤怒', '开心', '厌恶', '中性'] # Renamed from labels
    class_correct = {label: 0 for label in emotion_labels}
    class_total = {label: 0 for label in emotion_labels}
else:
    positive_class_correct = 0
    positive_class_total = 0
    negative_class_correct = 0
    negative_class_total = 0

stop_token_ids = []
if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
    if processor.tokenizer.eos_token_id is not None:
        stop_token_ids.append(processor.tokenizer.eos_token_id)


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=100, # Adjusted from 64, similar to qwen_audio_infer's max_length behavior.
    stop_token_ids=stop_token_ids
)

# 阶段1: 准备所有样本的输入数据
print("阶段1: 准备所有样本的输入数据...")
for i, sample in tqdm(enumerate(test_data), desc="准备输入数据", total=len(test_data)):
    try:
        true_label = sample["messages"][1]["content"]
        audio_assets = []
        conversation_content = []
        display_audio_rel_path = ""

        if args.task == 'speaker_recognition':
            audio_rel_path_1 = sample["audios"][0]
            audio_full_path_1 = os.path.join(audio_base_dir, audio_rel_path_1)
            # librosa.load might raise an exception if file is corrupted or not found
            audio_data_1, sr_1 = librosa.load(audio_full_path_1, sr=processor.feature_extractor.sampling_rate)
            audio_assets.append((audio_data_1, sr_1))


            audio_rel_path_2 = sample["audios"][1]
            audio_full_path_2 = os.path.join(audio_base_dir, audio_rel_path_2)
            audio_data_2, sr_2 = librosa.load(audio_full_path_2, sr=processor.feature_extractor.sampling_rate)
            audio_assets.append((audio_data_2, sr_2))
            
            display_audio_rel_path = f"{audio_rel_path_1}, {audio_rel_path_2}"
            conversation_content = [
                {"type": "text", "text": "音频1"},
                {"type": "audio", "audio_url": audio_full_path_1},
                {"type": "text", "text": "音频2"},
                {"type": "audio", "audio_url": audio_full_path_2},
                {"type": "text", "text": base_prompt_text},
            ]
        else: # deepfake or emotion
            audio_rel_path = sample["audios"][0]
            audio_full_path = os.path.join(audio_base_dir, audio_rel_path)
            audio_data, sr = librosa.load(audio_full_path, sr=processor.feature_extractor.sampling_rate)
            audio_assets.append((audio_data, sr))
            display_audio_rel_path = audio_rel_path
            conversation_content = [
                {"type": "audio", "audio_url": audio_full_path},
                {"type": "text", "text": base_prompt_text},
            ]

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": conversation_content},
        ]

        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False 
        )
        
        all_inputs_for_vllm.append({"prompt": text_prompt, "multi_modal_data": {"audio": audio_assets}})
        all_true_labels_for_batch.append(true_label)
        all_display_audio_paths_for_batch.append(display_audio_rel_path)
        all_original_samples_for_batch.append(sample) # Keep original sample for context if needed

    except Exception as e:
        print(f"准备样本 '{sample.get('audios', 'Unknown')}' 时出错: {e}")
        # Decide how to handle: skip, or add placeholder. For now, we skip and it won't be in batch.
        # If skipped, the total_predictions for accuracy calculation needs to be based on successful inferences.
        # Or, add to results with error status immediately.
        error_display_path = "Unknown"
        if args.task == 'speaker_recognition' and "audios" in sample and len(sample["audios"]) == 2:
            error_display_path = f"{sample['audios'][0]}, {sample['audios'][1]}"
        elif "audios" in sample and len(sample["audios"]) == 1:
            error_display_path = sample['audios'][0]
        
        results.append({
            "audio_url": error_display_path,
            "true_label": sample.get("messages", [{}, {"content": "Unknown"}])[1].get("content", "Unknown"),
            "predicted_label": "Error in preprocessing",
            "raw_prediction": str(e)
        })
        true_labels_list.append(sample.get("messages", [{}, {"content": "ErrorPlaceholder"}])[1].get("content", "ErrorPlaceholder"))
        pred_labels_list.append("Error")
        total_predictions += 1 # Count errored preprocessed samples as processed for overall count
        # Note: This sample won't be sent to vLLM. The final accuracy will reflect this.
        continue

# 阶段2: 执行批量推理
print(f"\n阶段2: 执行批量推理 (共 {len(all_inputs_for_vllm)} 个有效样本)...")
if not all_inputs_for_vllm:
    print("没有有效样本可供推理。")
    vllm_outputs = []
else:
    vllm_outputs = llm.generate(
        all_inputs_for_vllm,
        sampling_params=sampling_params,
    )

# 阶段3: 处理推理结果并评估
print("\n阶段3: 处理推理结果并评估...")
# total_predictions will now be the count of successfully preprocessed + errored preprocessed samples
# correct_predictions will be based on successfully inferred samples

# Iterate through the vLLM outputs and corresponding original data
for i, output_group in tqdm(enumerate(vllm_outputs), desc="处理推理结果", total=len(vllm_outputs)):
    # Each output_group in vllm_outputs corresponds to an item in all_inputs_for_vllm
    # We need to get the corresponding true_label and display_audio_rel_path
    true_label = all_true_labels_for_batch[i]
    display_audio_rel_path = all_display_audio_paths_for_batch[i]
    # original_sample = all_original_samples_for_batch[i] # If needed for more context

    try:
        predicted_text = output_group.outputs[0].text.strip()
        
        predicted_label = ""
        if args.task == 'deepfake' or args.task == 'speaker_recognition':
            if "是" in predicted_text and "否" not in predicted_text:
                 predicted_label = "是"
            elif "否" in predicted_text and "是" not in predicted_text:
                 predicted_label = "否"
            else:
                 if "是" in predicted_text: predicted_label = "是"
                 elif "否" in predicted_text: predicted_label = "否"
                 else: 
                    predicted_label = "否" # Fallback，打印错误
                    print(f"没匹配文本: {predicted_text}")

        else:  # emotion
            found_label = False
            for label_option in emotion_labels:
                if label_option in predicted_text:
                    predicted_label = label_option
                    found_label = True
                    break
            if not found_label:
                predicted_label = "中性"
                print(f"没匹配文本: {predicted_text}")
        
        results.append({
            "audio_url": display_audio_rel_path,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "raw_prediction": predicted_text
        })
        
        true_labels_list.append(true_label)
        pred_labels_list.append(predicted_label)
        
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
        
        total_predictions += 1 # Increment for successfully processed vLLM output
        
        # Print progress for this stage of processing results
        if (i + 1) % 10 == 0 or (i + 1) == len(vllm_outputs):
            # Accuracy here is based on samples that reached vLLM and are now processed
            # The overall accuracy will be correct_predictions / total_predictions (which includes preprocessing errors)
            current_processed_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"已处理 {total_predictions}/{len(test_data)} (总计) / {i+1}/{len(vllm_outputs)} (推理后) 样本。当前准确率 (基于已处理总数): {current_processed_acc:.4f}")

    except Exception as e:
        print(f"处理推理结果 '{display_audio_rel_path}' 时出错: {e}")
        results.append({
            "audio_url": display_audio_rel_path,
            "true_label": true_label,
            "predicted_label": "Error in postprocessing",
            "raw_prediction": str(e)
        })
        true_labels_list.append(true_label)
        pred_labels_list.append("Error")
        total_predictions += 1 # Count as processed for overall count
        continue

# 计算总体准确率
# The total_predictions already includes samples that errored during preprocessing or postprocessing.
# correct_predictions only includes successfully inferred and correct samples.
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"\n总体准确率 (vLLM): {accuracy:.4f} ({correct_predictions}/{total_predictions})")


# 根据任务类型生成报告
# Ensure report reflects the total number of samples attempted, including those with errors.
report = {
    "总尝试样本数": len(test_data), # Total samples from the input file
    "成功预处理并进入推理的样本数": len(all_inputs_for_vllm),
    "成功完成推理和后处理的样本数": len(vllm_outputs), # Number of samples for which vLLM produced an output
    "总计有效预测数 (用于指标计算)": total_predictions, # This is the count of samples for which we have a true_label and a (potentially 'Error') pred_label
    "正确预测数": correct_predictions,
    "总体准确率": accuracy,
}

if args.task == 'deepfake' or args.task == 'speaker_recognition':
    positive_label_name = "AI生成" if args.task == 'deepfake' else "同一说话人"
    negative_label_name = "真实音频" if args.task == 'deepfake' else "不同说话人"

    positive_accuracy = positive_class_correct / positive_class_total if positive_class_total > 0 else 0
    negative_accuracy = negative_class_correct / negative_class_total if negative_class_total > 0 else 0
    
    print(f"{positive_label_name}的准确率: {positive_accuracy:.4f} ({positive_class_correct}/{positive_class_total})")
    print(f"{negative_label_name}的准确率: {negative_accuracy:.4f} ({negative_class_correct}/{negative_class_total})")
    
    # Ensure true_positives etc. are calculated based on collected pred_labels_list and true_labels_list
    # This is more robust if errors occurred or if default predictions changed counts
    actual_true_positives = 0
    actual_false_positives = 0
    actual_false_negatives = 0
    actual_true_negatives = 0

    for true, pred in zip(true_labels_list, pred_labels_list):
        if true == "ErrorPlaceholder" or pred == "Error": continue # Skip errored samples for metric calculation
        if true == "是" and pred == "是":
            actual_true_positives += 1
        elif true == "否" and pred == "是":
            actual_false_positives += 1
        elif true == "是" and pred == "否":
            actual_false_negatives += 1
        elif true == "否" and pred == "否":
            actual_true_negatives += 1

    precision = actual_true_positives / (actual_true_positives + actual_false_positives) if (actual_true_positives + actual_false_positives) > 0 else 0
    recall = actual_true_positives / (actual_true_positives + actual_false_negatives) if (actual_true_positives + actual_false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    
    print("\n混淆矩阵:")
    print(f"真实\\预测 | 是 ({positive_label_name}) | 否 ({negative_label_name})")
    print(f"是 ({positive_label_name}) | {actual_true_positives:12d} | {actual_false_negatives:14d}")
    print(f"否 ({negative_label_name})| {actual_false_positives:12d} | {actual_true_negatives:14d}")
    
    report.update({
        f"{positive_label_name}_样本数": positive_class_total, # This remains based on initial counts from successfully processed samples
        f"{positive_label_name}_正确预测数": positive_class_correct,
        f"{positive_label_name}_准确率": positive_accuracy,
        f"{negative_label_name}_样本数": negative_class_total,
        f"{negative_label_name}_正确预测数": negative_class_correct,
        f"{negative_label_name}_准确率": negative_accuracy,
        "精确率 (recalced)": precision,
        "召回率 (recalced)": recall,
        "F1分数 (recalced)": f1_score,
        "混淆矩阵 (recalced)": {
            "真正例(TP)": int(actual_true_positives),
            "假正例(FP)": int(actual_false_positives),
            "假负例(FN)": int(actual_false_negatives),
            "真负例(TN)": int(actual_true_negatives)
        }
    })
    
elif args.task == 'emotion':
    emotion_accuracies = {}
    # Filter out errored labels for classification report
    filtered_true_labels = [tl for tl, pl in zip(true_labels_list, pred_labels_list) if tl != "ErrorPlaceholder" and pl != "Error"]
    filtered_pred_labels = [pl for tl, pl in zip(true_labels_list, pred_labels_list) if tl != "ErrorPlaceholder" and pl != "Error"]

    for label_option_report in emotion_labels:
        # Recalculate class_correct and class_total based on filtered lists
        current_class_total = filtered_true_labels.count(label_option_report)
        current_class_correct = sum(1 for true, pred in zip(filtered_true_labels, filtered_pred_labels) if true == label_option_report and pred == label_option_report)
        
        if current_class_total > 0:
            acc = current_class_correct / current_class_total
            emotion_accuracies[label_option_report] = acc
            print(f"{label_option_report}的准确率: {acc:.4f} ({current_class_correct}/{current_class_total})")
        else:
            emotion_accuracies[label_option_report] = 0
            print(f"{label_option_report}的准确率: 0.0000 (0/0)")

    class_report_dict = classification_report(filtered_true_labels, filtered_pred_labels, labels=emotion_labels, output_dict=True, zero_division=0)
    print("\n分类报告 (vLLM):")
    print(classification_report(filtered_true_labels, filtered_pred_labels, labels=emotion_labels, zero_division=0))
    
    cm = confusion_matrix(filtered_true_labels, filtered_pred_labels, labels=emotion_labels)
    print("\n混淆矩阵 (vLLM):")
    print("预测 →")
    header = "真实 ↓  " + " ".join([f"{label_cm:<8s}" for label_cm in emotion_labels]) # Adjusted spacing
    print(header)
    for i, label_row in enumerate(emotion_labels):
        row_str = f"{label_row:<6s} " + " ".join([f"{cm[i, j]:<8d}" for j in range(len(emotion_labels))]) # Adjusted spacing
        print(row_str)
    
    report.update({
        "各情绪类别准确率": emotion_accuracies,
        "分类报告": class_report_dict,
        "混淆矩阵": cm.tolist(), # cm is already a list of lists if tolist() was called. Ensure it's serializable.
        "原始混淆矩阵 (numpy)": cm.tolist() # Explicitly save as list for JSON
    })

# 保存详细结果
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"详细结果已保存到: {results_path}")

# 保存评估报告
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"评估报告已保存到: {report_path}")

