import torch
import torch.nn as nn

from LLM import MultiHeadSelfAttenion, RMSNorm, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.rmsnorm_1 = RMSNorm.RMSNorm(self.d_model, device=device)
        self.rmsnorm_2 = RMSNorm.RMSNorm(self.d_model, device=device)

        self.MHSA = MultiHeadSelfAttenion.MultiHeadSelfAttention(d_model=self.d_model, num_heads=self.num_heads, max_seq_len=self.max_seq_len, device=device)
        self.SwiGLU = SwiGLU.SwiGLU(d_model=self.d_model, d_ff=self.d_ff, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_x = self.rmsnorm_1(x)
        attended_x = self.MHSA(normalized_x)
        z = x + attended_x

        normalized_z = self.rmsnorm_2(z)
        attended_z = self.SwiGLU(normalized_z)
        output = z + attended_z
        return output
