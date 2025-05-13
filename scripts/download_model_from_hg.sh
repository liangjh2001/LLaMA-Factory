export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --token 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 --resume-download BAAI/bge-large-en-v1.5 --local-dir /data/liangjh/model_set/bge-large-en-v1.5 --local-dir-use-symlinks False --exclude "pytorch_model.bin" "*onnx*"
# --include "pytorch_model.bin"
# --exclude "consolidated.safetensors"