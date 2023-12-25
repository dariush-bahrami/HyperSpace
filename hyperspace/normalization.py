import torch
from torch import nn


class VectorNormalizer(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.batch_norm = nn.BatchNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=None,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )

    def mean(self):
        return self.batch_norm.running_mean

    def var(self):
        return self.batch_norm.running_var

    def std(self):
        return self.var().sqrt()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.batch_norm(input)
        output *= 1 / (self.num_features**0.5)
        return output
