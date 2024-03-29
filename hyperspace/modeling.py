import torch
from torch import nn

from .datatypes import DigitizationResult, HyperSpaceResult
from .digitazation import (
    digitize_vectors,
    get_reference_directions,
    get_reference_magnitudes,
)
from .normalization import VectorNormalizer


class HyperSpace(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_magnitude_subsections: int,
        num_direction_bisections: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_magnitude_subsections = num_magnitude_subsections
        self.num_direction_bisections = num_direction_bisections
        self.register_buffer(
            "reference_magnitudes",
            get_reference_magnitudes(num_magnitude_subsections),
        )
        self.register_buffer(
            "reference_directions",
            get_reference_directions(num_features, num_direction_bisections),
        )
        self.vector_normalizer = VectorNormalizer(num_features, eps=eps)

        self.register_buffer(
            "counts",
            torch.zeros(
                self.reference_magnitudes.shape[0],
                self.reference_directions.shape[0],
                dtype=torch.int64,
            ),
        )

    @property
    def probabilities(self) -> torch.Tensor:
        return self.counts / max(1, self.counts.sum())  # avoid divide by zero

    def sample(
        self,
        num_samples: int,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        indices = torch.multinomial(
            self.probabilities.view(-1),
            num_samples,
            replacement=True,
            generator=generator,
        )
        directions_indices = indices // self.probabilities.shape[1]
        magnitudes_indices = indices % self.probabilities.shape[1]
        directions = self.reference_directions[directions_indices]
        magnitudes = self.reference_magnitudes[magnitudes_indices]
        samples = directions * magnitudes.unsqueeze(1)
        samples = samples * self.vector_normalizer.std() + self.vector_normalizer.mean()
        return samples

    def update_stats(self, vectors: torch.Tensor) -> None:
        if self.training:
            self.vector_normalizer(vectors)
        else:
            raise RuntimeError("HyperSpace must be in training mode to update stats")

    def update_counts(self, vectors: torch.Tensor) -> DigitizationResult:
        if self.training:
            vectors = self.vector_normalizer(vectors)
            digitization_result = digitize_vectors(
                vectors,
                self.reference_magnitudes,
                self.reference_directions,
            )

            magnitude_indices, _, direction_indices, _ = digitization_result
            for i, j in zip(magnitude_indices, direction_indices):
                self.counts[i, j] += 1

            return digitization_result
        else:
            raise RuntimeError("HyperSpace must be in training mode to update counts")

    def digitize(self, vectors: torch.Tensor) -> DigitizationResult:
        vectors = self.vector_normalizer(vectors)
        return digitize_vectors(
            vectors,
            self.reference_magnitudes,
            self.reference_directions,
        )

    def forward(self, vectors: torch.Tensor) -> HyperSpaceResult:
        vectors = self.vector_normalizer(vectors)

        magnitude_indices, _, direction_indices, _ = digitize_vectors(
            vectors,
            self.reference_magnitudes,
            self.reference_directions,
        )

        space_counts = self.counts
        non_zero_space_counts = space_counts[space_counts > 0]
        total_counts = non_zero_space_counts.sum()
        batch_counts = self.counts[magnitude_indices, direction_indices]
        probabilities = batch_counts / max(1, total_counts)

        ranks = []
        for c in batch_counts:
            cumulative_counts = (
                non_zero_space_counts[non_zero_space_counts <= c]
            ).sum()
            rank = (cumulative_counts - (0.5 * c)) / max(1, total_counts)
            ranks.append(rank)
        ranks = torch.stack(ranks)

        return HyperSpaceResult(
            counts=batch_counts,
            probabilities=probabilities,
            ranks=ranks,
        )
