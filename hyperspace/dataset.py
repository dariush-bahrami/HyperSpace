import torch
from torch.utils.data import Dataset


class VectorDatasetFromTensor(Dataset):
    def __init__(self, vectors: torch.Tensor) -> None:
        assert vectors.ndim == 2
        self.vectors = vectors

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.vectors[index]
