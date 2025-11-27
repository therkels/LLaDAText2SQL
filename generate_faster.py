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


# --- OPTIMIZED ---
# Replaced the for-loop with a vectorized implementation
def get_num_transfer_tokens_vectorized(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True) # Shape: (batch_size, 1)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create a range tensor: [0, 1, 2, ..., steps-1]
    # Shape: (1, steps)
    step_indices = torch.arange(steps, device=mask_index.device).unsqueeze(0)

    # Compare remainder with step_indices.
    # Thanks to broadcasting, this creates a (batch_size, steps) boolean mask.
    # Where step_indices < remainder, it's True (so we add 1).
    # Shape: (batch_size, steps)
    add_one = (step_indices < remainder)

    # Add the base amount (broadcasted) to the boolean mask
    # True becomes 1, False becomes 0
    num_transfer_tokens = base + add_one
    
    return num_transfer_tokens.to(torch.int64)
# --- END OPTIMIZED ---


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L). B is batch size.
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
        
        # --- OPTIMIZED ---
        # Call the new vectorized function
        num_transfer_tokens = get_num_transfer_tokens_vectorized(block_mask_index, steps)
        # --- END OPTIMIZED ---
        
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

            # --- OPTIMIZED ---
            # Replaced the for-loop over the batch size with a fully vectorized implementation
            
            # Get the k value for this step for each item in the batch
            # Shape: (batch_size)
            current_k_values = num_transfer_tokens[:, i]

            # Get the indices that would sort the confidences in descending order
            # Shape: (batch_size, seq_len)
            sorted_indices = torch.argsort(confidence, dim=-1, descending=True)

            # Create a range tensor: [0, 1, 2, ..., seq_len-1]
            # Shape: (1, seq_len)
            seq_range = torch.arange(x0.shape[1], device=x0.device).unsqueeze(0)

            # Broadcasted comparison:
            # Create a (batch_size, seq_len) boolean mask where seq_range < current_k_values
            # This gives us True for the top 'k' positions for each row.
            # current_k_values.unsqueeze(-1) has shape (batch_size, 1)
            # seq_range has shape (1, seq_len)
            # Result 'top_k_mask' has shape (batch_size, seq_len)
            top_k_mask = (seq_range < current_k_values.unsqueeze(-1))

            # We now have a mask of the *ranks* (e.g., "select 1st, 2nd, 3rd")
            # We need a mask of the *original positions*.
            # torch.scatter_ will "un-sort" the mask for us.
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index.scatter_(dim=1, index=sorted_indices, src=top_k_mask)
            
            # Apply the update in one go
            x[transfer_index] = x0[transfer_index]
            # --- END OPTIMIZED ---

    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
               "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
               "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

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
    import time
    start_time = time.time()
    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    end_time = time.time()
    print(f"Generation took {end_time - start_time:.2f} seconds")
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()