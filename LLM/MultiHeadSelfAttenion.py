import torch
import torch.nn as nn
from einops import rearrange

from LLM.Linear import Linear
from LLM.ScaledDotProductAttention import ScaledDotProductAttention
from LLM.RoPE import RoPE

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int,  device: torch.device):
        super().__init__()
        self.d_model = d_model # dimensionality of the TransformerBlock input (d_model => num_heads * head_dim)
        self.num_heads = num_heads # number of heads to use in multi-head self attention
        self.head_dim = self.d_model // self.num_heads
        
        self.wq = Linear(input_dim=self.d_model, output_dim=self.d_model, device = device)
        self.wk = Linear(input_dim=self.d_model, output_dim=self.d_model, device = device)
        self.wv = Linear(input_dim=self.d_model, output_dim=self.d_model, device = device)
        self.wo = Linear(input_dim=self.d_model, output_dim=self.d_model, device = device)
        
        self.sdpa = ScaledDotProductAttention()
        self.rope = RoPE(theta=10000, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "batch seq_len (num_head d_k) -> batch num_head seq_len d_k", num_head = self.num_heads, d_k = self.head_dim)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "batch num_head seq_len d_k -> batch seq_len (num_head d_k)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape

        query = self._split_heads(self.wq(x)) # (B, H, Q, D)
        key = self._split_heads(self.wk(x)) # (B, H, k, D)
        value = self._split_heads(self.wv(x)) # (B, H, k, D)

        query = rearrange(query, "b h q d -> (b h) q d")
        key = rearrange(key, "b h k d -> (b h) k d")
        value = rearrange(value, "b h k d -> (b h) k d")

        query = self.rope(query)
        key = self.rope(key)
        context, attn = self.sdpa(query, key, value, mask = True)

        context = rearrange(context, "(b h) q d -> b h q d", h = self.num_heads)
        context = self._combine_heads(context)
        output = self.wo(context)

        return output
