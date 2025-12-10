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
    cache_dir = ""
    tokenizer = AutoTokenizer("Qwen/Qwen2.5-7B-Instruct", trust_remote_code = True)
    model = LLM(model="Qwen/Qwen2.5-7B-Instruct")