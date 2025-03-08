import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.toolkit.utils import apply_rotary_pos_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, theta=500000.0, scaling_factor=8.0):
        super().__init__()
        self.theta = theta
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cached", None)
        self.register_buffer("sin_cached", None)

    def _update_cos_sin_cache(self, seq_len):
        required_len = max(seq_len, self.max_seq_len)
        if self.cos_cached is None or required_len > self.cos_cached.shape[1]:
            t = (
                torch.arange(required_len, device=self.inv_freq.device)
                / self.scaling_factor
            )
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
            self.register_buffer("sin_cached", emb.sin()[None, :, None, :])
            self.max_seq_len = required_len

    def forward(self, x, seq_len):
        self._update_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :seq_len, :, :].to(x.device),
            self.sin_cached[:, :seq_len, :, :].to(x.device),
        )


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_kv_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_size // num_heads

        self.q_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.k_proj = nn.Linear(embed_size, self.head_dim * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(embed_size, self.head_dim * num_kv_heads, bias=False)
        self.o_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.rotary = RotaryPositionalEmbedding(
            self.head_dim, theta=10000.0, scaling_factor=1.0
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary(x, seq_len)
        Q = apply_rotary_pos_emb(Q, cos, sin)
        K = apply_rotary_pos_emb(K, cos, sin)

        # Grouped query attention
        K = K.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        V = V.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return self.o_proj(attn_output.view(batch_size, seq_len, -1))


class SwiGLU(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.w1 = nn.Linear(embed_size, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, embed_size, bias=False)
        self.w3 = nn.Linear(embed_size, hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
