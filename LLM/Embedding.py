import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dims: int, device: torch.device, dtype: str = None):
        # num_embeddings -> size of the vocabulary
        # embedding_dim -> dimensions of the embedding vector
        super().__init__()
        self.embedding_table = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dims, device=device, dtype=dtype)))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_table[token_ids]