import torch
import torch.nn as nn

from LLM.Linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device):
        super().__init__()
        self.d_model = d_model # # dimensionality of the TransformerBlock input (d_model => num_heads * head_dim)
        self.d_ff = d_ff # represents the size of position-wise feed-forward network. Usually greater than d_model, eg., 3072
        # the model architecture goes like this: d_model -> linear.shape(d_ff) -> linear.shape(d_model)
        # d_ff must be 8/3 * d_model

        self.w1 = Linear(d_model, d_ff, device)
        self.w2 = Linear(d_ff, d_model, device)
        self.w3 = Linear(d_model, d_ff, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size,   seq_len, d_model)
        w1_out = self.w1(x) # (b, s, d_ff)
        SiLU = w1_out * torch.sigmoid(w1_out) # (b, s, d_ff)
        GLU = SiLU * self.w3(x) # (b, s, d_ff)
        output = self.w2(GLU) # (b, s, d_model)
        return output