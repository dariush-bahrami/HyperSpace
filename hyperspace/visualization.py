import matplotlib.pyplot as plt
import torch

from .modeling import HyperSpace


def visualize_hyper_space(hyper_space: HyperSpace) -> tuple[plt.Figure, plt.Axes]:
    num_magnitude_tokens = hyper_space.reference_magnitudes.shape[0]
    num_direction_tokens = hyper_space.reference_directions.shape[0]
    counts = hyper_space.counts
    r, theta = (counts > 0).nonzero().unbind(dim=1)
    r = r / num_magnitude_tokens
    theta = theta / num_direction_tokens
    min_theta = 0
    max_theta = 2 * torch.pi
    theta = theta * (max_theta - min_theta) + min_theta
    min_r = hyper_space.reference_magnitudes.min()
    max_r = hyper_space.reference_magnitudes.max()
    r = r * (max_r - min_r) + min_r

    r_mean = hyper_space.vector_normalizer.mean().norm()
    r_std = hyper_space.vector_normalizer.std().norm()
    r = r * r_std + r_mean

    colors = theta
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    ax.scatter(theta, r, s=16, c=colors, cmap="prism", alpha=0.5)
    ax.set_rticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return fig, ax
