import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, device: torch.device, dtype: str = None):
        super().__init__()
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(output_dim, input_dim, device=device, dtype=dtype)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_transpose = rearrange(self.weight, "... o i -> ... i o")
        dot_product = einsum(x, weight_transpose, "... i, ... i o -> ... o")
        return dot_product