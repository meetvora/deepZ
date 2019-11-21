import torch
import torch.nn as nn


def verify(zonotope_predictions: torch.Tensor, label: int) -> bool:
    # TODO: Improve inspection approach.
    # Currently compares max & min possible activations (naive).

    noise_magnitude = torch.abs(zonotope_predictions[1:]).sum(dim=0)
    min_scores, max_scores = (zonotope_predictions[0] - noise_magnitude, zonotope_predictions[0] + noise_magnitude)
    max_scores[label - 1] = min_scores[label - 1] - 1
    return torch.all(min_scores[label - 1] > max_scores)
