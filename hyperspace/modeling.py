import torch
from torch import nn

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
    ):
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
        self.vector_normalizer = VectorNormalizer(num_features)

        self.register_buffer(
            "counts",
            torch.zeros(
                self.reference_directions.shape[0],
                self.reference_magnitudes.shape[0],
                dtype=torch.int64,
            ),
        )

    @property
    def probabilities(self):
        return self.counts / max(1, self.counts.sum())  # avoid divide by zero

    def sample(self, num_samples: int, generator=None):
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

    @property
    def threshold(self):
        return self.probabilities[self.probabilities > 0].min()

    def update_stats(self, vectors: torch.Tensor):
        if self.training:
            self.vector_normalizer(vectors)
        else:
            raise Exception("HyperSpace must be in training mode to update stats")

    def update_counts(self, vectors):
        if self.training:
            vectors = self.vector_normalizer(vectors)
            digitization_result = digitize_vectors(
                vectors,
                self.reference_magnitudes,
                self.reference_directions,
            )
            direction_indices = digitization_result["direction_indices"]
            magnitude_indices = digitization_result["magnitude_indices"]

            for i, j in zip(direction_indices, magnitude_indices):
                self.counts[i, j] += 1
        else:
            raise Exception("HyperSpace must be in training mode to update counts")

    def forward(self, vectors):
        vectors = self.vector_normalizer(vectors)

        digitization_result = digitize_vectors(
            vectors,
            self.reference_magnitudes,
            self.reference_directions,
        )
        direction_indices = digitization_result["direction_indices"]
        magnitude_indices = digitization_result["magnitude_indices"]

        probabilities = self.probabilities[direction_indices, magnitude_indices]
        return probabilities
