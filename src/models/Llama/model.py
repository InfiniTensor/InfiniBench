import torch
import torch.nn as nn
from src.ops.torch.ops import *


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(
            config["hidden_size"],
            config["num_attention_heads"],
            config["num_key_value_heads"]
        )
        self.ffn = SwiGLU(config["hidden_size"], config["intermediate_size"])
        self.norm1 = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.norm2 = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x


class Llama3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        if self.config.get("scale_embeddings", False):
            x *= math.sqrt(self.config["hidden_size"])
        
        seq_len = input_ids.shape[1]
        # Create causal mask and combine with padding mask
        causal_mask = torch.full((seq_len, seq_len), float('-inf'), dtype=x.dtype, device=x.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        # Expand mask for batch and heads: (batch_size, 1, seq_len, seq_len)
        causal_mask = causal_mask[None, None, :, :]
        
        if attention_mask is not None:
            # Convert padding mask to additive form: 0 for valid, -inf for padding
            padding_mask = torch.where(attention_mask.bool(), 0.0, float('-inf'))
            mask = causal_mask + padding_mask[:, None, None, :]
        else:
            mask = causal_mask
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        return self.lm_head(x)

