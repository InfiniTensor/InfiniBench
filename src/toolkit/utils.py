import torch
import torch.nn.functional as F

def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


def apply_repetition_penalty(logits, input_ids, repetition_penalty=1.0):
    if repetition_penalty != 1.0:
        prev_tokens = input_ids[0].tolist()
        for token_id in set(prev_tokens):
            # Scale logits for repeated tokens
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

    return logits


def apply_length_penalty(logits, input_ids, tokenizer, initial_prompt_length, length_penalty=0.0):
    if length_penalty != 0.0:
        eos_token_id = tokenizer.eos_token_id
        current_length = input_ids.size(1)
        generated_length = current_length - initial_prompt_length
        penalty = generated_length * length_penalty
        logits[0, :] -= penalty  # Penalize all tokens
        logits[0, eos_token_id] += penalty  # Compensate EOS token

    return logits


def apply_topp(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')

    return logits