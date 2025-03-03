#!/bin/sh


conda activate sglang

config_path=config/qwen-token-retrieval.yaml
output_dir=../result_release/infinitbench/qwen-token-retrieval

world_size=8
datasets="passkey,number_string,kv_retrieval,longdialogue_qa_eng,math_find,code_debug"
pkill pt_main_thread
bash scripts/multiprocessing-benchmark.sh  \
    --config_path $config_path \
    --datasets $datasets \
    --output_dir_path $output_dir
python benchmark/merge.py \
    --output_dir_path ${output_dir} \
    --datasets ${datasets} \
    --world_size ${world_size}
python benchmark/infinitebench_eval.py --result-dir ${output_dir}

world_size=4
datasets="longbook_sum_eng,longbook_choice_eng,longbook_qa_eng"
pkill pt_main_thread
bash scripts/multiprocessing-benchmark-tp2.sh  \
    --config_path $config_path \
    --datasets $datasets \
    --output_dir_path $output_dir
python benchmark/merge.py \
    --output_dir_path ${output_dir} \
    --datasets ${datasets} \
    --world_size ${world_size}
python benchmark/infinitebench_eval.py --result-dir ${output_dir}