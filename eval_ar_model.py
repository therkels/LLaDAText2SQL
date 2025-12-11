'''
Much of the LLVM setup was referenced from CSE595 HW 4. Used much of the tempalte code to configure the model.
'''


import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import json
import re
import pandas as pd
# from dataset.synthetic_text_to_sql.get_raw_data import get_raw_data, convert_data_to_namedtuples
import time
import os
from datasets import load_from_disk
import argparse

from llvm import LLM, SamplingParams


def create_prompt(context: str, instruction: str):
    prompt = f"""
        You are a senior analyst who is an expert in SQL query generation.
        Given a schema and prompt, generate the SQL.
        ## Rules for generation
        1. **Wrap the SQL with <sql>...</sql> tags.** Only the first SQL block will be considered.
        2. **If using name aliasing, use column_x, where x is a integer**.
        ## Schema:
        {context}
        ## Prompt:
        {instruction}
    """
    return prompt

def parse_sql(output):
    print(f"----output----\n{output}\n--------------")
    xml_pattern = re.compile(r"<sql>(.*?)</sql>", re.DOTALL)
    md_pattern = re.compile(r"```sql(.*?)```", re.DOTALL)

    def first_in(s: str):
        m = xml_pattern.search(s)
        if m:
            return m.group(1).strip()
        m = md_pattern.search(s)
        if m:
            return m.group(1).strip()
        return None

    if isinstance(output, str):
        return first_in(output)

    # assume list/tuple of strings
    r = [first_in(s) for s in output]
    return r

def main():
    parser = argparse.ArgumentParser(
        description='Score essays using vLLM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output files
    parser.add_argument(
        '--input-file',
        type=str,
        default='train_dev.csv',
        help='Path to input CSV file with essays'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default='essay_scores.jsonl',
        help='Path to output JSONL file with results'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of essays to process (for testing)'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate predictions against ground truth scores'
    )

    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model name or path (e.g., "meta-llama/Llama-2-7b-chat-hf", "Qwen/Qwen2.5-7B-Instruct")'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Directory to cache models (default: HuggingFace cache)'
    )

    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=None,
        help='Number of GPUs for tensor parallelism (default: auto-detect)'
    )

    # Processing settings
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Number of essays to process in each batch (default: 100)'
    )

    parser.add_argument(
        '--use-chat-template',
        action='store_true',
        default=True,
        help='Use chat template formatting (for chat models)'
    )

    parser.add_argument(
        '--no-chat-template',
        dest='use_chat_template',
        action='store_false',
        help='Disable chat template (for base/completion models)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature (0.0 = deterministic, higher = more random)'
    )

    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling parameter'
    )

    args = parser.parse_args()
    CACHE_DIR = '/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/vllm_cache'
    tokenizer_kwargs = {'trust_remote_code': True,
                        'cache_dir': CACHE_DIR}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        **tokenizer_kwargs
    )

    available_gpus = torch.cuda.device_count()
    tensor_parallel_size = args.tensor_parallel_size or available_gpus

    if tensor_parallel_size > available_gpus:
        print(f"Warning: Requested {tensor_parallel_size} GPUs but only {available_gpus} available")
        tensor_parallel_size = available_gpus

    print(f"Available GPUs: {available_gpus}")
    print(f"Using tensor_parallel_size: {tensor_parallel_size}")

    # Configure vLLM LLM instance
    model_kwargs = {
        'model': args.model_name,
        'tensor_parallel_size': tensor_parallel_size
    }
    model_kwargs['download_dir'] = CACHE_DIR

    print("Loading model (this may take a few minutes for large models)...")
    model = LLM(**model_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=256,  # or set dynamically if needed
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # Load input data (CSV with columns: context, instruction)
    df = pd.read_csv(args.input_file)
    if args.limit:
        df = df.head(args.limit)

    results = []
    for idx, row in df.iterrows():
        context = row.get('context', '')
        instruction = row.get('instruction', '')
        prompt = create_prompt(context, instruction)
        outputs = model.generate([prompt], sampling_params)
        sql = parse_sql(outputs[0].outputs[0].text)
        results.append({
            'idx': idx,
            'context': context,
            'instruction': instruction,
            'sql': sql
        })
        print(f"Instance {idx}:\nPrompt: {prompt}\nSQL: {sql}\n---")

    # Save results to output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')