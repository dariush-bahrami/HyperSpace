from typing import Optional

from torch.utils.data import DataLoader, Dataset

from .modeling import HyperSpace


def evaluate_using_just_positive_data_loader(
    model: HyperSpace,
    positive_data_loader: DataLoader,
) -> dict:
    model.eval()

    threshold = model.threshold

    num_true_positives = 0
    num_false_negatives = 0
    for batch in positive_data_loader:
        output = model(batch)
        num_true_positives += (output >= threshold).sum()
        num_false_negatives += (output < threshold).sum()

    accuracy = num_true_positives / (num_true_positives + num_false_negatives)

    return {"accuracy": accuracy}


def evaluate_using_positive_and_negative_data_loaders(
    model: HyperSpace,
    positive_data_loader: DataLoader,
    negative_data_loader: Dataset,
) -> dict:
    model.eval()
    threshold = model.threshold
    num_true_positives = 0
    num_false_negatives = 0
    for batch in positive_data_loader:
        output = model(batch)
        num_true_positives += (output >= threshold).sum()
        num_false_negatives += (output < threshold).sum()

    num_true_negatives = 0
    num_false_positives = 0
    for batch in negative_data_loader:
        output = model(batch)
        num_true_negatives += (output < threshold).sum()
        num_false_positives += (output >= threshold).sum()

    precision = num_true_positives / (num_true_positives + num_false_positives)
    recall = num_true_positives / (num_true_positives + num_false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    total = (
        num_true_positives
        + num_false_positives
        + num_true_negatives
        + num_false_negatives
    )
    correct = num_true_positives + num_true_negatives
    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    model: HyperSpace,
    positive_data_loader: DataLoader,
    negative_data_loader: Optional[DataLoader] = None,
) -> dict:
    if negative_data_loader is None:
        return evaluate_using_just_positive_data_loader(model, positive_data_loader)
    else:
        return evaluate_using_positive_and_negative_data_loaders(
            model, positive_data_loader, negative_data_loader
        )
