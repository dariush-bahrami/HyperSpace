import torch
from torch.utils.data import Dataset


class VectorDataset(Dataset):
    def __init__(self, vectors: torch.Tensor):
        assert vectors.ndim == 2
        self.vectors = vectors

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, idx):
        return self.vectors[idx]
