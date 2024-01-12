from typing import NamedTuple

import torch


class DigitizationResult(NamedTuple):
    magnitude_indices: torch.Tensor
    magnitude_offsets: torch.Tensor
    direction_indices: torch.Tensor
    direction_offsets: torch.Tensor


class HyperSpaceResult(NamedTuple):
    counts: torch.Tensor
    probabilities: torch.Tensor
    ranks: torch.Tensor
