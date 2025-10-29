import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    mask_id = 126336 # The LLaDA mask token ID

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

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
    text_to_sql_prompt = """Generate the SQL for this

Schema:
CREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');

Question:
What is the total volume of timber sold by each salesperson, sorted by salesperson?"""

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
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
