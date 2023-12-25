import torch


@torch.jit.script
def get_reference_directions(num_features: int, num_bisections: int) -> torch.Tensor:
    vectors = []
    # add basis vectors
    for i in range(num_features):
        v = torch.zeros(num_features)
        v[i] = 1
        vectors.append(v)

    # add negative of basis vectors
    for i in range(num_features):
        v = torch.zeros(num_features)
        v[i] = -1
        vectors.append(v)

    # add bisectors
    for i in range(num_bisections):
        temp = []
        vectors.append(vectors[0])
        for v1, v2 in zip(vectors[:-1], vectors[1:]):
            temp.append(v1)
            bisector = v1 + v2  # formula for unit vector bisector
            bisector_norm = torch.norm(bisector, p=2)
            bisector = bisector / bisector_norm
            temp.append(bisector)
        vectors = temp

    # stack vectors into matrix
    vectors = torch.stack(vectors, dim=0)

    return vectors


@torch.jit.script
def get_reference_magnitudes(num_subsections: int) -> torch.Tensor:
    steps = num_subsections
    spaces = torch.linspace(1 / steps, 1 - 1 / steps, steps)
    return torch.cat(((1 - spaces).flip(0), torch.ones(1), 1 + spaces))


@torch.jit.script
def digitize_vectors(
    vectors: torch.Tensor,
    reference_magnitudes: torch.Tensor,
    reference_directions: torch.Tensor,
) -> dict:
    magnitudes = torch.norm(vectors, p=2, dim=1, keepdim=True)
    magnitude_offsets, magnitude_indices = (
        (magnitudes - reference_magnitudes).abs().min(dim=1)
    )
    unit_vectors = vectors / magnitudes
    direction_offsets, direction_indices = (unit_vectors @ reference_directions.T).max(
        dim=1
    )
    return {
        "magnitude_offsets": magnitude_offsets,
        "magnitude_indices": magnitude_indices,
        "direction_offsets": direction_offsets,
        "direction_indices": direction_indices,
    }
