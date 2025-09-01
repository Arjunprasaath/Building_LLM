import torch
import torch.nn as nn

from LLM import Embedding, RMSNorm, Linear, TransformerBlock

class LanguageModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, device: torch.device):
        super().__init__()
        self.vocab_size = vocab_size # number of unique items in the output vocabulary to predict (size of vocabulary)
        self.context_length = context_length # maximum number of token to process at once
        self.d_model = d_model # dimensionality of model embedding and sublayers
        self.num_layers = num_layers # number of transformer blocks to use
        self.num_heads = num_heads # number of heads to use multi-headed attention. d_model must be evenly divisible by num_heads
        self.d_ff = d_ff # dimensionality of the feed-forward inner layers
        self.device = device

        self.embedding = Embedding.Embedding(num_embeddings=self.vocab_size, embedding_dims=self.d_model, device=self.device)
        self.rmsnorm = RMSNorm.RMSNorm(d_model=self.d_model, device=self.device)
        self.final_layer = Linear.Linear(input_dim=self.d_model, output_dim=self.vocab_size, device=self.device)
        self.transformer = nn.ModuleList([TransformerBlock.TransformerBlock(d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff, max_seq_len=self.context_length, theta=10000, device=self.device) for _ in range(self.num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.transformer:
            x = block(x)
        
        normalized_x = self.rmsnorm(x)
        final_linear = self.final_layer(normalized_x)
        return final_linear
            
