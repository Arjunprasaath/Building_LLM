import math
import torch
import torch.nn as nn
from einops import rearrange, einsum

from LLM.utils import Softmax

class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = Softmax(dim = -1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: bool) -> torch.tensor:
        """
        Computes attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        query: (batch_size, seq_len, d_q/ d_k)
        key: (batch_size, seq_len, d_k)
        value: (batch_size, seq_len, d_v / d_k)
        mask: (True/ False)
        """
        b, q, d_k = query.shape
        _, k, _ = key.shape

        scores = einsum(query, key, "b q d_k, b k d_k -> b q k")
        scores = scores / math.sqrt(d_k)
        
        if mask:
            casual_mask = torch.triu(torch.ones(q, k, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(casual_mask, float('-inf'))
        attention = self.softmax(scores) # (b q k)
        output = einsum(attention, value, "b q k, b k d_k -> b q d_k")
        return output, attention
