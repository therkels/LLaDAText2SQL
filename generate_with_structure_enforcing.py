import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import json
import re
import pandas as pd


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
def generate(model, input_ids, attention_mask=None, steps=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    (MODIFIED)
    Args:
        model: Mask predictor.
        input_ids: A tensor of shape (1, L) that INCLUDES the [MASK] tokens for in-filling.
        steps: Sampling steps.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = input_ids.clone()
    prompt_index = (x != mask_id) # Index of all non-mask tokens (the "prompt")

    # This is the "in-filling" block, which finds all masks
    block_mask_index = (x == mask_id)
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
        else:
            raise NotImplementedError(remasking)

        # We no longer need the block-specific masking, as we fill all masks at once
        # x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]

    return x

@ torch.no_grad()
def generate_original(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
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

class SpiderDataset(torch.utils.data.Dataset):
    '''
    Dataset class for Spider dataset that preloads all data into memory and prepares prompts for LLM.
    '''
    def __init__(self, path_to_json, path_to_documents, tokenizer, device='cuda', 
                 gen_length_query=512, gen_length_explanation=512):
        self.path_to_documents = path_to_documents
        self.tokenizer = tokenizer
        self.device = device
        self.gen_length_query = gen_length_query
        self.gen_length_explanation = gen_length_explanation
        self.mask_id = 126336  # The LLaDA mask token ID
        
        # Set up tokenizer
        if tokenizer.padding_side != 'left':
            tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token_id != self.mask_id, "Pad token ID cannot be the same as Mask ID"
        
        # Load all data at initialization
        with open(path_to_json, 'r') as f:
            self.data = [json.loads(line) for line in f]
            
        # Preload all document contexts
        self.contexts = {}
        for entry in self.data:
            print(entry.get('external_knowledge', 'No external_knowledge field found'))
            doc_name = entry['external_knowledge']
            if doc_name not in self.contexts:
                with open(f"{path_to_documents}/{doc_name}", 'r') as doc_f:
                    self.contexts[doc_name] = doc_f.read()
                    
        # Pre-tokenize the JSON template parts
        self.template_parts = {
            'start': tokenizer('{\n  "query": "', add_special_tokens=False, return_tensors="pt").input_ids.to(device),
            'middle': tokenizer('",\n  "explanation": "', add_special_tokens=False, return_tensors="pt").input_ids.to(device),
            'end': tokenizer('"\n}', add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        }
    
    def prepare_prompt(self, context, instruction):
        """Prepare the prompt with proper formatting and tokenization."""
        # Format the text-to-SQL prompt
        text_to_sql_prompt = f"Generate the SQL for this\n\nSchema:\n{context}\n\nQuestion:\n{instruction}"
        
        # Apply chat template
        messages = [{"role": "user", "content": text_to_sql_prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Tokenize the prompt
        prefix_tokens = self.tokenizer(
            formatted_prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Create mask tensors
        masks1 = torch.full(
            (1, self.gen_length_query), 
            self.mask_id, 
            dtype=torch.long, 
            device=self.device
        )
        masks2 = torch.full(
            (1, self.gen_length_explanation), 
            self.mask_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        # Concatenate all parts
        input_ids = torch.cat([
            prefix_tokens,
            self.template_parts['start'],
            masks1,
            self.template_parts['middle'],
            masks2,
            self.template_parts['end']
        ], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return input_ids, attention_mask
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        context = self.contexts[entry['external_knowledge']]
        input_ids, attention_mask = self.prepare_prompt(context, entry['instruction'])
        
        return {
            'instance_id': entry['instance_id'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'raw_instruction': entry['instruction'],
            'raw_context': context
        }

def preprocess_spider(path_to_json, path_to_documents, path_to_output_sql, batch_size=32):
    '''
    Preprocess the Spider dataset to generate SQL queries using the LLaDA model.
    Args:
        path_to_json: Path to the Spider JSONL file.
        path_to_documents: Path to the folder containing database schema documents.
        path_to_output_sql: Path to the folder to save generated SQL queries.
        batch_size: Batch size for the dataloader.
    '''
    # Create dataset
    dataset = SpiderDataset(path_to_json, path_to_documents, tokenizer=AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True))
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Process batches
    for batch in dataloader:
        for instance_id, instruction, context in zip(
            batch['instance_id'], 
            batch['instruction'], 
            batch['context']
        ):
            print(f"Processing instance {instance_id}")
            print("Context:", context)
            print("Instruction:", instruction)
            print("-" * 80)
        break  # Remove this break when ready to process all data

    print(len(dataset))
    
def generate_spider_sql(path_to_json, path_to_documents, path_to_output_sql):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    with open(path_to_json, 'r') as f:
        data = [json.loads(line) for line in f]

    for entry in data:
        instance_id = entry['instance_id']
        instruction = entry['instruction']
        context = entry['external_knowledge']
        # context holds the name of the document file in path_to_documents
        with open(f"{path_to_documents}/{context}", 'r') as doc_f:
            context = doc_f.read()
        # generate from context and instruction
        output = text_to_sql(model, tokenizer, context, instruction)
        print(f"Generated output for instance {instance_id}:\n{output[0]}\n")
        # extract sql from output
        parsed_output = json.loads(extract_first_json(output[0]))
        sql_query = parsed_output.get("query", "")
        # save to submission folder
        print(f"Generated SQL query for instance {instance_id}:\n{sql_query}\n")
        with open(f"{path_to_output_sql}/{instance_id}.sql", 'w') as out_f:
            out_f.write(sql_query)

def parse_sql(output):
    pat = re.compile(r"<sql>(.*?)</sql>", re.DOTALL)

    def first_in(s: str):
        m = pat.search(s)
        return m.group(1) if m else None

    if isinstance(output, str):
        return first_in(output)

    # assume list/tuple of strings
    for s in output:
        r = first_in(s)
        if r is not None:
            return r
    return None

def generate_eval_sql(ids, contexts, prompts, model=None, tokenizer=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None or tokenizer is None:
        model = AutoModel.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    df = pd.DataFrame(columns=["id", "out_sql"])

    for _id, context, prompt in zip(ids, contexts, prompts):
        out_list = text_to_sql(model, tokenizer, context, prompt) 
        sql = parse_sql(out_list)
        df.loc[len(df)] = [_id, sql]

    return df

def text_to_sql_structured(model, tokenizer, context, instruction, gen_length_query=512, gen_length_explanation=512):
    mask_id = 126336 # The LLaDA mask token ID
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        print("Setting tokenizer padding side to 'left'")
        tokenizer.padding_side = 'left'

    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != mask_id, "Pad token ID cannot be the same as Mask ID (126336)"

    # --- This is the new prompt format for your text-to-SQL task ---
    # Combine the system instruction, schema, and question into one string.
    text_to_sql_prompt = f"Generate the SQL for this\n\nSchema:\n{context}\n\nQuestion:\n{instruction}"

    # --- Construct the full input with the JSON template and masks ---
    
    # 1. Format the user prompt with the Instruct model's chat template
    print("Applying chat template...")
    messages = [{"role": "user", "content": text_to_sql_prompt}]
    # We tokenize=False because we'll tokenize the whole thing at once
    formatted_prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # 2. Tokenize the prompt prefix
    prefix_tokens = tokenizer(formatted_prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # 3. Define generation lengths for query and explanation
    gen_length_query = 512
    gen_length_explanation = 512 # Total ~250 tokens, similar to your example
    
    # 4. Create the template parts and mask tensors
    part1 = tokenizer('{\n  "query": "', add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    masks1 = torch.full((1, gen_length_query), mask_id, dtype=torch.long, device=device)
    
    part2 = tokenizer('",\n  "explanation": "', add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    masks2 = torch.full((1, gen_length_explanation), mask_id, dtype=torch.long, device=device)
    
    part3 = tokenizer('"\n}', add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # 5. Concatenate all parts to form the final input_ids
    input_ids = torch.cat([prefix_tokens, part1, masks1, part2, masks2, part3], dim=1)
    
    # 6. Create a full attention mask
    attention_mask = torch.ones_like(input_ids)

    print(f"Input tensor shape (with masks): {input_ids.shape}")
    print("-------------------------------------------\n")

    print("Generating SQL and Explanation (in-filling)...")
    # Call the modified generate function
    # We no longer pass gen_length or block_length
    out = generate(model, input_ids, attention_mask, steps=128, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=mask_id)
    
    print("\n--- Generated Output (Full) ---")
    # Decode the *entire* output tensor to see the filled-in template
    output = tokenizer.batch_decode(out, skip_special_tokens=True)
    return output

def text_to_sql(model, tokenizer, context, instruction, gen_length=256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompts = [f"""Given a schema and prompt, generate the SQL. 
                  Wrap your SQL in between <sql> and </sql> tags. 
                  Only the SQL in between first set of sql tags will be considered. 
                  \n\nSchema:\n{context}\n\nPrompt:\n{instruction}"""]
                  
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
    out = generate_original(model, input_ids, attention_mask, steps=128, gen_length=gen_length, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    print(output)
    parsed_sql = parse_sql(output)
    return parsed_sql


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    output = text_to_sql(
        model, 
        tokenizer, 
        context="CREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');",
        instruction="What is the total volume of timber sold by each salesperson, sorted by salesperson?"
    )
    print(output)

if __name__ == '__main__':
    # main()
    # t2s_instruction = '/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/Spider2/spider2-snow/spider2-snow.jsonl'
    # document_context = '/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/Spider2/spider2-snow/resource/documents/'
    # # Oh yeah, doing some local saves so nobody cheats :)
    # output_sql_folder = '/home/CSE595/LLaDATextToSQL/spider_sql_outputs/'
    # # generate_spider_sql('/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/Spider2/spider2-snow/spider2-snow.jsonl', 
    # #                     '/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/Spider2/spider2-snow/resource/documents/',
    # #                     '/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/Spider2/spider2-snow/evaluation_suite/submission_folder/')
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True)
    # preprocess_spider(t2s_instruction, document_context, output_sql_folder)
    main()