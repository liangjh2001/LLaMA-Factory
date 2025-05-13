from datasets import DatasetDict, load_dataset, concatenate_datasets
import os
import json

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
gsm8k = load_dataset("parquet",
                    data_files="/data/liangjh/model_set/datasets/gsm8k/main/test-00000-of-00001.parquet",
                    split='train', cache_dir="/data/liangjh/model_set/datasets")
arc_challenge = load_dataset("parquet",
                    data_files="/data/liangjh/model_set/datasets/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet",
                    split='train', cache_dir="/data/liangjh/model_set/datasets")
truthful_qa = load_dataset("parquet",
                    data_files="/data/liangjh/model_set/datasets/truthful_qa/multiple_choice/validation-00000-of-00001.parquet",
                    split='train', cache_dir="/data/liangjh/model_set/datasets")
hellaswag = load_dataset("arrow",
                    data_files="/home/yanghh/.cache/huggingface/datasets/hellaswag/default/0.1.0/512a66dd8b1b1643ab4a48aa4f150d04c91680da6a4096498a5e5f799623d5ae/hellaswag-test.arrow",
                    split='train', cache_dir="/data/liangjh/model_set/datasets")
winogrande = load_dataset("arrow",
                    data_files="/home/yanghh/.cache/huggingface/datasets/winogrande/winogrande_xl/1.1.0/a826c3d3506aefe0e9e9390dcb53271070536586bab95849876b2c1743df56e2/winogrande-test.arrow",
                    split='train', cache_dir="/data/liangjh/model_set/datasets")
mmlu = load_dataset("parquet",
                    data_files="/data/liangjh/model_set/datasets/mmlu/all/test-00000-of-00001.parquet",
                    split='train', cache_dir="/data/liangjh/model_set/datasets")



mt_bench = []  # 用于存储 JSON 对象的列表

# 逐行读取JSONL文件
with open("/data/liangjh/FastChat-main/fastchat/llm_judge/data/mt_bench/question.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        # 解析JSON对象
        json_object = json.loads(line)
        # 将JSON对象存储到列表中
        mt_bench.append(json_object)

# 打印存储的JSON对象列表
for question in mt_bench:
    print(question)

print(gsm8k[0])