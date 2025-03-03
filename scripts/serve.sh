#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=Qwen/Qwen2-7B-Instruct
SERVER_PORT=62726

python benchmark/serve.py \
    --model-path ${MODEL_PATH} \
    --dp 1 \
    --port ${SERVER_PORT} \
    --disable-cuda-graph \
    --disable-regex-jump-forward \
    --disable-radix-cache \
    --max-running-requests 1 \
    --mem-fraction-static 0.85 \
    --context-length 1048576 \
    --sgl-conf-file config/qwen-token-retrieval.yaml

# # serve longer context using tp
# export CUDA_VISIBLE_DEVICES="0,1"
# python benchmark/serve.py \
#     --model-path ${MODEL_PATH} \
#     --dp 1 \
#     --tp 2 \
#     --port ${SERVER_PORT} \
#     --disable-cuda-graph \
#     --disable-regex-jump-forward \
#     --disable-radix-cache \
#     --mem-fraction-static 0.85 \
#     --context-length 1048576 \
#     --sgl-conf-file config/llama-token-retrieval.yaml