"""Standard self-attention with the released FlashAttention checkpoint key layout.

Uses PyTorch SDPA instead of changing Tabula's backbone to additive FastFormer.
"""
import torch
from torch import nn
from torch.nn import functional as F


class NativeMHA(nn.Module):
    def __init__(self, dim, heads, dropout, bias):
        super().__init__()
        if dim % heads:
            raise ValueError('Attention width must be divisible by the head count')
        self.Wqkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.heads, self.dropout = heads, dropout

    def forward(self, x):
        batch, length, dim = x.shape
        qkv = self.Wqkv(x).reshape(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0., is_causal=False)
        return self.out_proj(out.transpose(1, 2).reshape(batch, length, dim))


class NativeFlashAttention(nn.Module):
    def __init__(self, d_token, n_heads, dropout, bias=True):
        super().__init__()
        self.attention = NativeMHA(d_token, n_heads, dropout, bias)

    def forward(self, x_q, x_kv):
        # The official FlashAttention wrapper also uses self-attention here.
        return self.attention(x_q)
