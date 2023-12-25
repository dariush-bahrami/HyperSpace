from torch.utils.data import DataLoader

from .modeling import HyperSpace


def train(model: HyperSpace, data_loader: DataLoader) -> None:
    model.train()

    # update stats
    for batch in data_loader:
        model.update_stats(batch)

    # update counts
    for batch in data_loader:
        model.update_counts(batch)
