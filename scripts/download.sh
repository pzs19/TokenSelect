mkdir benchmark/data
mkdir benchmark/data/infinite-bench
mkdir benchmark/data/longbench

python benchmark/download.py

cd benchmark/data/infinite-bench

for file in code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey; do
    echo "Downloading ${file}.jsonl"
    aria2c -x 16 "https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/${file}.jsonl" -o ./${file}.jsonl
done