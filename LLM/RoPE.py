import torch
import torch.nn as nn
from einops import rearrange, einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device):
        super().__init__()
        assert d_k % 2 == 0, "Embedding dimension must be even"
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len # maximum sequence length that will be inputted

        # precompute inverse frequency
        half_dim = self.d_k // 2
        freq_seq = torch.arange(half_dim)
        inv_freq = 1.0 / (self.theta ** (freq_seq / half_dim))
        self.register_buffer("inv_freq", inv_freq)


        # precompute sin and cos values
        token_pos = torch.arange(max_seq_len)
        angles = einsum(token_pos, inv_freq, "... i, ... j -> ... i j")
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        self.register_buffer("sin_cached", sin)
        self.register_buffer("cos_cached", cos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        # token_position = torch.arange(seq_len, dtype=torch.long)
        # token_position = rearrange(token_position, "s -> 1 s")

        x1 = x[..., ::2] # even
        x2 = x[..., 1::2] # odd

        sin = self.sin_cached[:seq_len, :].unsqueeze(0)
        cos = self.cos_cached[:seq_len, :].unsqueeze(0)

        rotated_x_even = x1 * cos - x2 * sin
        rotated_x_odd = x1 * sin + x2 * cos
        x_out = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        x_out = rearrange(x_out, "... half_dim i -> ... (half_dim i)")
        return x_out
