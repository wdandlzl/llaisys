import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def is_root():
    try:
        return llaisys.Distributed.rank() == 0
    except Exception:
        return True


def root_print(*args, **kwargs):
    if is_root():
        print(*args, **kwargs)


def resolve_model_path(model_path=None):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        root_print(f"Loading model from local path: {model_path}")
        return model_path

    raise FileNotFoundError(
        f"--model must be a real local directory, got: {model_path}"
    )


def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_hf_model(model_path, device_name="cpu"):
    tokenizer = load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )
    return tokenizer, model


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def load_llaisys_model(model_path, device_name):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip_hf", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    llaisys.Distributed.init()
    try:
        model_path = resolve_model_path(args.model)
        tokenizer = load_tokenizer(model_path)

        tokens = None
        output = None

        if is_root() and not args.skip_hf:
            hf_tokenizer, hf_model = load_hf_model(model_path, args.device)

            start_time = time.time()
            tokens, output = hf_infer(
                args.prompt,
                hf_tokenizer,
                hf_model,
                max_new_tokens=args.max_steps,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            end_time = time.time()

            del hf_model
            gc.collect()

            root_print("\n=== Answer ===\n")
            root_print("Tokens:")
            root_print(tokens)
            root_print("\nContents:")
            root_print(output)
            root_print("\n")
            root_print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

        llaisys.Distributed.barrier()

        model = load_llaisys_model(model_path, args.device)

        start_time = time.time()
        llaisys_tokens, llaisys_output = llaisys_infer(
            args.prompt,
            tokenizer,
            model,
            max_new_tokens=args.max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        end_time = time.time()

        root_print("\n=== Your Result ===\n")
        root_print("Tokens:")
        root_print(llaisys_tokens)
        root_print("\nContents:")
        root_print(llaisys_output)
        root_print("\n")
        root_print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

        if args.test and is_root() and (tokens is not None):
            assert llaisys_tokens == tokens
            root_print("\033[92mTest passed!\033[0m\n")
    finally:
        try:
            llaisys.Distributed.finalize()
        except Exception:
            pass

if args.test and is_root() and (tokens is not None):
    assert llaisys_tokens == tokens
    root_print("\033[92mTest passed!\033[0m\n")