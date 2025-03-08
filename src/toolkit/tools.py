import argparse
import importlib
import json
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from transformers import AutoTokenizer

from .utils import (
    apply_length_penalty,
    apply_repetition_penalty,
    apply_topp,
    color_text,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Universal Model Test Framework")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time,",
        required=False,
        help="The input prompt",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose output"
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature parameter for sampling"
    )
    parser.add_argument("--topk", type=float, help="Top-k sampling parameter")
    parser.add_argument("--topp", type=float, help="Top-p sampling parameter")
    parser.add_argument("--length_penalty", type=float, help="Length penalty parameter")
    parser.add_argument(
        "--repetition_penalty", type=float, help="Repetition penalty parameter"
    )
    return parser.parse_args()


def dynamic_import(class_path):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_yaml_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    weight_base = config["weights"]["base_path"]
    config["weights"]["checkpoint_paths"] = [
        weight_base + file for file in config["weights"]["files"]
    ]

    if not os.path.exists(config["tokenizer"]["path"]):
        raise ValueError(f"Tokenizer path invalid: {config['tokenizer']['path']}")

    return config


def load_model_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    return {
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config.get("num_key_value_heads"),
        "intermediate_size": config["intermediate_size"],
        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
    }


def load_model(model_class: nn.Module, checkpoint_paths: List[str], config):
    print(color_text("- Loading model...", 32))
    model = model_class(config)
    state_dict = {}
    for path in checkpoint_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    converted_state_dict = {}
    for key, value in state_dict.items():
        # Convert key names
        new_key = key
        new_key = new_key.replace("model.embed_tokens", "embedding")
        new_key = new_key.replace("model.layers", "layers")
        new_key = new_key.replace("self_attn", "attention")

        # Handle MLP conversion
        if "mlp" in new_key:
            new_key = new_key.replace("mlp.gate_proj", "ffn.w1")
            new_key = new_key.replace("mlp.up_proj", "ffn.w3")
            new_key = new_key.replace("mlp.down_proj", "ffn.w2")

        # Handle norms
        new_key = new_key.replace("input_layernorm", "norm1")
        new_key = new_key.replace("post_attention_layernorm", "norm2")
        new_key = new_key.replace("model.norm", "norm")

        # Final lm_head
        new_key = new_key.replace("lm_head", "lm_head")

        converted_state_dict[new_key] = value

    # Load state_dict with strict=False to ignore missing buffers
    model.load_state_dict(converted_state_dict, strict=False)
    print(color_text("- Model loaded.", 32))
    return model


def load_tokenizer(tokenizer_path):
    print(color_text("- Loading tokenizer...", 32))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(color_text("- Tokenizer loaded.", 32))
    return tokenizer


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=2048,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.0,
    length_penalty=0.02,
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
        model.device
    )  # Shape: [1, seq_len]
    initial_prompt_length = input_ids.size(1)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)  # Shape: [1, seq_len, vocab_size]

        # Focus on last token's logits
        logits = logits[:, -1, :]  # Shape: [1, vocab_size]
        logits = logits / temperature

        # Apply repetition penalty
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)

        # Length penalty: Encourage EOS as generated sequence grows
        logits = apply_length_penalty(
            logits, input_ids, tokenizer, initial_prompt_length, length_penalty
        )

        # Top-p sampling
        logits = apply_topp(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: [1, 1]

        if next_token.item() == tokenizer.eos_token_id:
            break

        # Decode the new token and yield
        decoded_token = tokenizer.decode(next_token[0], skip_special_tokens=True)
        yield decoded_token

        # Append new token (maintains 2D shape)
        input_ids = torch.cat([input_ids, next_token], dim=-1)


def infer(model, tokenizer, prompt):
    print(color_text("- Start inference...", 32))
    print(f"{prompt}", end="", flush=True)

    # Print each token immediately
    for token in generate_text(model, tokenizer, prompt):
        print(token, end="", flush=True)
    print()
