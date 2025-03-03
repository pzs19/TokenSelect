import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import torch

import requests
from transformers import AutoTokenizer

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from datasets import load_from_disk
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tp_size", type=int, default=1)
    args, extra_args = parser.parse_known_args()
    config = OmegaConf.load(args.config_path)
    cli_config = OmegaConf.from_cli(extra_args)
    config = OmegaConf.merge(config, cli_config)
    config.output_dir_path = args.output_dir_path
    config.rank = args.rank
    config.world_size = args.world_size
    config.tp_size = args.tp_size
    config.verbose = args.verbose
    if not hasattr(config.model, "tokenizer_path"):
        config.model.tokenizer_path = config.model.path
    if not hasattr(config, "truncation"):
        config.truncation = None

    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    config.datasets = []
    for d in datasets_list:
        config.datasets.append(d.strip())
    return config


def get_model_and_tokenizer(config, kernel_size):
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)

    if config.model.type == "token-retrieval":
        from patcher.token_retrieval import patch
        patch(
            rope_base=config.model.rope_base,
            rope_scale=config.model.rope_scale,
            rope_model="ROPE_LLAMA",
            max_n_tokens=config.model.max_n_tokens,
            n_init=config.model.n_init,
            n_local=config.model.n_local,
            top_k=config.model.top_k,
            kernel_size=kernel_size,
        )
    else:
        raise NotImplementedError()

    from sglang.srt.server import Runtime

    model = Runtime(
        model_path=config.model.path,
        dtype=config.dtype,
        chunked_prefill_size=config.chunk_size,
        max_prefill_tokens=config.max_len,
        mem_fraction_static=0.85,
        disable_cuda_graph=True,
        disable_regex_jump_forward=True,
        disable_radix_cache=True,
        disable_disk_cache=True,
        disable_flashinfer_sampling=True,
        max_running_requests=1,
        tp_size=config.tp_size,
        port=60000 + (config.rank+1) * 10,
        additional_ports=list(range(60001, 60100)),
        context_length=config.max_len,
    )

    return model, tokenizer


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    model_name = model_name.strip().lower()
    if model_name == "vicuna":
        from fastchat.conversation import get_conv_template

        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif model_name in ["mistral-inst", "qwen", "minicpm", "llama-3-inst", "yi"]:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        raise NotImplementedError

    return prompt


def load_infinite_bench(path, data_name) -> str:
    import re

    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """
    print(f"read {data_name}.jsonl")
    fin = open(os.path.join(path, data_name + ".jsonl"), "r")
    lines = fin.readlines()
    fin.close()
    data = [json.loads(line) for line in lines]

    def get_answer(inp: dict):
        if data_name in ["code_debug", "longbook_choice_eng"]:
            OPTIONS = "ABCD"
            if isinstance(inp["answer"], str):
                ret = [inp["answer"], OPTIONS[inp["options"].index(inp["answer"])]]
            elif isinstance(inp["answer"], list):
                if len(inp["answer"]) == 1:
                    ret = [
                        inp["answer"][0],
                        OPTIONS[inp["options"].index(inp["answer"][0])],
                    ]
                elif len(inp["answer"]) == 2 and inp["answer"][1] in [
                    "A",
                    "B",
                    "C",
                    "D",
                ]:
                    ret = inp["answer"]
                else:
                    raise ValueError
            else:
                raise ValueError
            return ret
        return inp["answer"]

    ret = []
    for eg in data:
        # ================= Code tasks
        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg["input"])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            instance = {"func": func, "func_call": func_call, "context": eg["context"]}
        elif data_name in ["code_debug", "code_debug_qa"]:
            # Load source code
            instance = {"context": eg["context"]}
            if data_name == "code_debug":
                instance.update(
                    {
                        "OPTION_A": eg["options"][0],
                        "OPTION_B": eg["options"][1],
                        "OPTION_C": eg["options"][2],
                        "OPTION_D": eg["options"][3],
                    }
                )
        # ================= Code tasks
        elif data_name == "longdialogue_qa_eng":
            instance = {"context": eg["context"]}
        # ==================== Long book tasks
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            instance = {"context": eg["context"]}
            if data_name == "longbook_choice_eng":
                instance.update(
                    {
                        "question": eg["input"],
                        "OPTION_A": eg["options"][0],
                        "OPTION_B": eg["options"][1],
                        "OPTION_C": eg["options"][2],
                        "OPTION_D": eg["options"][3],
                    }
                )
            elif data_name in ["longbook_qa_eng", "longbook_qa_chn"]:
                instance.update(
                    {
                        "question": eg["input"],
                    }
                )
        elif data_name == "math_calc":
            instance = {"context": eg["context"]}
        elif data_name == "math_find":
            prompt = eg["input"]
            context = eg["context"]
            # Find "the * number" from the prompt
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            # Replace the number with the answer
            prefix = f"What is {target_number} in the following list?"
            instance = {"prefix": prefix, "context": context, "input": prompt}
        elif data_name == "kv_retrieval":
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
                "key": eg["input"][6:44],
            }
            assert eg["input"][6] == '"'
            assert eg["input"][43] == '"'
        else:
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
            }
        ans = get_answer(eg)
        instance["answers"] = ans if isinstance(ans, list) else [ans]
        instance["length"] = len(instance["context"].split())
        instance["all_classes"] = None

        ret.append(instance)
        # if len(ret) > 4:
        #     break
    return ret


def post_process(pred, model_name, dataset):
    if model_name == "qwen":
        pred = pred.split("<|im_end|>")[0]

    if dataset == "samsum":
        pred = pred.split("\n")[0].strip()

    return pred


def get_pred(
        model,
        tokenizer,
        data,
        max_length,
        max_gen,
        prompt_format,
        dataset,
        model_name,
        truncation: str = None,
        rank: int = None,
        world_size: int = None,
        verbose: bool = False,
        out_path: str = ""
):
    preds = []
    data = list(data)
    if os.path.exists(out_path):
        with open(out_path, 'r') as f: 
            for line in f.readlines():
                preds.append(json.loads(line.strip()))

    if world_size is not None:
        data = data[rank::world_size]
    data = data[len(preds):]
    
    cur = 0
    total = len(data)
            
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        end_token_ids = [tokenizer.eos_token_id]
        if model_name == "llama-3-inst":
            end_token_ids.append(
                tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
            )

        if model_name == "qwen":
            end_token_ids.append(
                tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
            )

        if dataset == "samsum":
            end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

            if model_name.strip().lower() in ["mistral-inst"]:
                add_special_tokens = False
            else:
                add_special_tokens = True

        else:
            add_special_tokens = True

        tokenized_prompt = tokenizer(
            prompt,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        ).input_ids[0]

        if truncation is None:
            if len(tokenized_prompt) > max_length - max_gen:
                if verbose:
                    print(f"Length {len(tokenized_prompt)}. Skipped.")
                continue

        else:
            if truncation == "suffix":
                length = len(tokenized_prompt)
                if length > max_length - max_gen:
                    if verbose:
                        print("over length")
                    init_token_num = 128
                    prompt = tokenizer.decode(
                        tokenized_prompt[:init_token_num].tolist()
                        + tokenized_prompt[
                          -(max_length - max_gen - init_token_num):
                          ].tolist()
                    )
                    tokenized_prompt = tokenizer(
                        prompt,
                        truncation=False,
                        return_tensors="pt",
                        add_special_tokens=add_special_tokens,
                    ).input_ids[0]
            else:
                raise NotImplementedError

        sampling_params = {
            "max_new_tokens": max_gen,
            "stop_token_ids": list(set(end_token_ids)),
            "temperature": 0,
        }

        output = requests.post(
            model.url + "/generate",
            json={
                "input_ids": tokenized_prompt.tolist(),
                "sampling_params": sampling_params,
            },
        )
        output = output.json()["text"]

        pred = post_process(output, model_name, dataset)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
                "token_length": len(tokenized_prompt) + max_gen,
            }
        )
        cur += 1
        if verbose:
            print(f"----------{cur}/{total}----------")
            print("Length: ", len(tokenized_prompt))
            print("Question:", prompt[-100:])
            print("Pred:", pred)
            print("Answer:", json_obj["answers"])
            print("")
    return preds


if __name__ == "__main__":
    config = parse_args()

    output_dir_path = config.output_dir_path

    datasets = config.datasets

    dataset_timing = {}

    dataset2prompt = json.load(open("benchmark/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/config/dataset2maxlen.json", "r"))
    if config.conv_type == "llama-3-inst":
        dataset2ks = json.load(open("benchmark/config/dataset2ks_llama.json", "r"))
    elif config.conv_type == "qwen":
        dataset2ks = json.load(open("benchmark/config/dataset2ks_qwen.json", "r"))

    multiprocessing = config.world_size is not None and config.world_size > 1
    if multiprocessing:
        assert config.rank in list(range(config.world_size))

    # predict on each dataset
    for dataset in datasets:
        start_time = time.time()
        dname = dataset
        if dataset in [
            "passkey",
            "number_string",
            "kv_retrieval",
            "longdialogue_qa_eng",
            "longbook_sum_eng",
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_qa_chn",
            "math_find",
            "math_calc",
            "code_run",
            "code_debug",
        ]:
            path = "../data/infinitebench"
            data = load_infinite_bench(path, dname)
        else:
            data = load_from_disk(f"../data/longbench/{dataset}")

        out_path = os.path.join(output_dir_path, f"{dname}.jsonl")
        if multiprocessing:
            out_path = out_path + f"_{config.rank}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        print(f"Pred {dname}")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        kernel_size = dataset2ks.get(dataset, 0)
        
        model, tokenizer = get_model_and_tokenizer(config, kernel_size)

        preds = get_pred(
            model,
            tokenizer,
            data,
            config.max_len,
            max_gen,
            prompt_format,
            dataset,
            config.conv_type,
            config.truncation,
            config.rank,
            config.world_size,
            config.verbose,
            out_path,
        )

        del model
        torch.cuda.empty_cache()

        end_time = time.time()
        elapsed_time = end_time - start_time
        dataset_timing[dataset] = elapsed_time

        with open(out_path, "w+", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")

    timing_file_path = os.path.join(output_dir_path, "dataset_timing.json")
    if multiprocessing:
        timing_file_path = timing_file_path + f"_{config.rank}"
    with open(timing_file_path, "a") as f:
        json.dump(dataset_timing, f, indent=4)