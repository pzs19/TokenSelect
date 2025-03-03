#!/bin/bash

SHORT=f:,d:,o:,h
LONG=config_path:,datasets:,output_dir_path:,help

PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    echo "Invalid Arguments."
    exit 2
fi
eval set -- "$PARSED"

world_size=8
# datasets="longbook_sum_eng,longbook_choice_eng,longbook_qa_eng"

while true; do
    case "$1" in
        -h|--help)
            echo "Usage: $0 [--config_path <file>] [--datasets <dataset_name>] [--output_dir_path <dir>] [--help]"
            exit
            ;;
        -f|--config_path)
            config_path="$2"
            shift 2
            ;;
        -d|--datasets)
            datasets="$2"
            shift 2
            ;;
        -o|--output_dir_path)
            output_dir_path="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

mkdir -p ${output_dir_path}
rm -rf ~/.cache/outlines
NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=0,1 python benchmark/pred.py \
    --config_path ${config_path} \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size} \
    --rank 0 \
    --tp_size 2 &
    echo "worker 0 started using CUDA device 01"
sleep 60
rm -rf ~/.cache/outlines
NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=2,3 python benchmark/pred.py \
    --config_path ${config_path} \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size} \
    --rank 1 \
    --tp_size 2 &
    echo "worker 1 started using CUDA device 23"
sleep 60
rm -rf ~/.cache/outlines
NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=4,5 python benchmark/pred.py \
    --config_path ${config_path} \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size} \
    --rank 2 \
    --tp_size 2 &
    echo "worker 2 started using CUDA device 45"
sleep 60
rm -rf ~/.cache/outlines
NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=6,7 python benchmark/pred.py \
    --config_path ${config_path} \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size} \
    --rank 3 \
    --tp_size 2 &
    echo "worker 3 started using CUDA device 67"

wait
echo done