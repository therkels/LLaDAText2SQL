import torch
import numpy as np
import pandas as pd
import time
import os
import json
import re
import argparse
from transformers import AutoTokenizer
from datasets import load_from_disk
from vllm import LLM, SamplingParams

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
    r = [first_in(s) for s in output]
    return r

def atomic_save(save_path, df):
    if not save_path: return
    tmp = save_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, save_path)

def main():
    parser = argparse.ArgumentParser(
        description='Score essays using vLLM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input-file', type=str, default='/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/test_data')
    parser.add_argument('--output-file', type=str, default='eval_results_ar.csv')
    parser.add_argument('--max_eval', type=int, default=20000)
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--tensor-parallel-size', type=int, default=None)
    parser.add_argument('--chunk-size', type=int, default=20) # Increased default chunk size for speed
    parser.add_argument('--use-chat-template', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--top-k', type=int, default=50)
    

    args = parser.parse_args()
    
    save_path = args.output_file
    autosave_every = 50
    
    CACHE_DIR = '/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/vllm_cache'
    
    # tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=CACHE_DIR)

    # GPU setup
    available_gpus = torch.cuda.device_count()
    tensor_parallel_size = args.tensor_parallel_size or available_gpus
    if tensor_parallel_size > available_gpus:
        tensor_parallel_size = available_gpus

    print(f"Available GPUs: {available_gpus}")
    print(f"Using tensor_parallel_size: {tensor_parallel_size}")

    # Model setup
    print("Loading model...")
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=tensor_parallel_size,
        download_dir=CACHE_DIR,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # Data setup
    arrow_dataset = load_from_disk(args.input_file)
    arrow_dataset = arrow_dataset.select(range(min(args.max_eval, len(arrow_dataset))))    
    
    results = []
    df_data = [] 
    
    chunk_size = args.chunk_size
    eval_count = 0

    try:
        # Loop over the dataset in chunks
        last_save_count = 0
        for i in range(0, len(arrow_dataset), chunk_size):
            
            batch = arrow_dataset[i : i + chunk_size]
            
            # HF Dataset slicing gives a dict of lists
            batch_contexts = batch['sql_context']
            batch_prompts = batch['sql_prompt']
            batch_ids = batch['id']
            
            formatted_prompts = []
            
            # Prepare prompts
            for context, instruction in zip(batch_contexts, batch_prompts):
                ar_prompt = create_prompt(context, instruction)
                messages = [{"role": "user", "content": ar_prompt}]
                
                formatted_input = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False, 
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted_input)
            
            time_start = time.time()
            outputs = model.generate(formatted_prompts, sampling_params)
            time_end = time.time()
            
            batch_time = time_end - time_start
            avg_time_per_item = batch_time / len(outputs) if len(outputs) > 0 else 0
            
            # Process outputs
            for j, output_item in enumerate(outputs):
                generated_text = output_item.outputs[0].text
                sql = parse_sql(generated_text)
                
                _id = batch_ids[j]
                context = batch_contexts[j]
                instruction = batch_prompts[j]
                
                results.append({
                    'idx': i + j,
                    'context': context,
                    'instruction': instruction,
                    'sql': sql,
                    'generated_text': generated_text
                })
                
                df_data.append([_id, sql, avg_time_per_item, generated_text])
                
                eval_count += 1
            
            # Autosave logic
            # We check if we crossed a multiple of autosave_every
            if save_path and (len(df_data) - last_save_count >= autosave_every):
                temp_df = pd.DataFrame(df_data, columns=["id", "out_sql", "time_taken", "generated_text"])
                atomic_save(save_path=save_path, df=temp_df)
                print(f"Saved {len(df_data)} records...")
                last_save_count = len(df_data)

            if eval_count >= args.max_eval:
                break
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    finally:
        if save_path:
            print("Saving final CSV results...")
            final_df = pd.DataFrame(df_data, columns=["id", "out_sql", "time_taken", "generated_text"])
            final_df.to_csv(save_path, index=False)

    # Save JSON results to a DIFFERENT file
    json_path = args.output_file.replace('.csv', '.jsonl')
    if json_path == args.output_file: json_path += ".jsonl" # fallback if input wasn't .csv

    print(f"Saving JSON results to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

if __name__ == "__main__":
    main()