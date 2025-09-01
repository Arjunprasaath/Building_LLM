import torch
import torch.nn as nn
from einops import reduce

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = 'cpu', dtype: float = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce((x ** 2), "... d_model -> ... 1", "mean") + self.eps)
        x_norm = x / rms
        output = (x_norm * self.weight).to(x_dtype)
        return output
