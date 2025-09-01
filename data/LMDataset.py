import torch
from torch.utils.data import Dataset
import numpy as np

class LMDataset(Dataset):
    def __init__(self, token_file, block_size) -> None:
        super().__init__()
        self.tokens = np.memmap(token_file, dtype=np.int32, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.tokens[idx: idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.tokens[idx + 1: idx + self.block_size + 1].astype(np.int64))
        return x, y
