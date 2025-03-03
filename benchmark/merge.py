import os
import argparse
from collections import defaultdict
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    args = parser.parse_args()
    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    datasets_list = [s.strip() for s in datasets_list]
    args.datasets = datasets_list
    return args

if __name__ == "__main__":
    args = parse_args()
    
    for dataset in args.datasets:

        out_path = os.path.join(
            args.output_dir_path,
            f"{dataset}.jsonl"
        )

        
        lines = []
        for rank in range(args.world_size):
            file_path = out_path + f"_{rank}"
            try:
                f = open(file_path, "r")
                lines += f.readlines()
                f.close()
            except Exception as e:
                print(e)
                continue

        lines = [l.strip() for l in lines]
        f = open(out_path, "w+")
        f.write(
            "\n".join(lines)
        )
        f.close()
    
    timing_all = defaultdict(float)
    out_path = os.path.join(
        args.output_dir_path,
        f"dataset_timing.json"
    )
    for rank in range(args.world_size):
        file_path = out_path + f"_{rank}"
        try:
            f = open(file_path, "r")
            timing_rank = json.loads(f.read())
            for dataset, time in timing_rank.items():
                timing_all[dataset] += time
        except Exception as e:
            print(e)
            continue
        
    json.dump(timing_all, open(out_path, "w+"))