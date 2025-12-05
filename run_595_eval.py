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
from strict_outputs.remask import Text2SQLMasker

def get_args():
    parser = argparse.ArgumentParser(description="Run LLaDA Text2SQL evaluation.")
    parser.add_argument('--use_dynamic_context', action='store_true', help='Enable dynamic context prediction')
    parser.add_argument('--remask_strategy', type=str, default='low_confidence', help='Masking strategy to use during generation')
    parser.add_argument('--max_eval', type=int, default=20, help='Maximum number of evaluations')
    return parser.parse_args()

#Dynamic context prediction
import dynamic_context.ContextPredictor as cp

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@ torch.no_grad()
def generate_original(model, tokenizer, prompt, text2sql_masker=None, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    start = time.time()
    # print(f"start: {time.time() - start}")
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'Text2SQL':
                if text2sql_masker is None:
                    raise ValueError("Text2SQL remasking requires a Text2SQLMasker instance.")

                x0_p = text2sql_masker.get_masking_confidence_scores(x0, tokenizer)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    # print(f"end: {time.time() - start}")
    return x

def extract_first_json(text):
    start = text.find('{')
    if start == -1:
        return None

    stack = []
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            stack.append(ch)
        elif ch == '}':
            stack.pop()
            if not stack:
                return text[start:i+1]  # return the full {...}
    return None  # if unmatched
    

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

def generate_eval_sql(dataset, args, model=None, tokenizer=None, device=None, batch_size=1, save_path=None, autosave_every=50):
    device = "cuda"
    # Setup dynamic context prediction
    if args.use_dynamic_context:
        SAVED_MODEL_PATH = "/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/saved_models/predict_model.pt"
        context_model = cp.ContextPredictor()
        context_model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
        context_model = context_model.to(device).eval()
        context_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if model is None or tokenizer is None:
        model = AutoModel.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    df = pd.DataFrame(columns=["id", "out_sql"])

    def atomic_save():
        if not save_path: return
        tmp = save_path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, save_path)

    try:
        eval_count = 0
        for i,instance in enumerate(dataset):
            eval_count += 1
            context = instance['sql_context']
            prompt = instance['sql_prompt']
            _id = instance['id']
            # Predict context length bucket
            if args.use_dynamic_context:
                context_length = cp.predict_context_length(context_model, context_tokenizer, context, prompt, device=device) + 10
            else:
                context_length = 256
            sql = text_to_sql(model, tokenizer, context, prompt, remask_strategy=args.remask_strategy, block_length=context_length, gen_length=context_length)
            df.loc[len(df)] = [_id, sql]
            if save_path and autosave_every and (i % autosave_every == 0):
                atomic_save()
            if (eval_count+1) % args.max_eval == 0:
                break
    except Exception as e:
        # catch-all for anything else
        print(f"Unexpected error: {e}")
        raise
    finally:
        if save_path:
            print("Saving results")
            df.to_csv(save_path, index=False)
    return df



def text_to_sql(model, tokenizer, context, instruction, remask_strategy, gen_length=256, block_length=32):
    device = 'cuda'
    # print("building prompts")
  # build flat prompts
    prompts = [
            "Given a schema and prompt, generate the SQL.\n"
            "Wrap your SQL in <sql>...</sql>.\n"
            "Only the first SQL block will be considered.\n\n"
            "Do Not use name aliasing in the SQL.\n\n"
            f"Schema:\n{context}\n\nPrompt:\n{instruction}"
    ]
    # print(prompts)
                  
    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)
    # print("Starting Generation")
    # print(f"gen length:{gen_length}, block_length:{block_length}")
    text2sql_masker = Text2SQLMasker()
    out = generate_original(model, tokenizer, input_ids, attention_mask, steps=128, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking=remask_strategy)
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    parsed_sql = parse_sql(output)
    return parsed_sql

def main():
    args = get_args()
    device = 'cuda'
    print(f"Using device: {device}")
    arrow_dataset = load_from_disk("/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/test_data")

    generate_eval_sql(arrow_dataset, args, save_path="eval.csv")


if __name__ == '__main__':
    main()