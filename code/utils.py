import torch
import torch.nn as nn


def verify(zonotope_predictions: torch.Tensor, label: int) -> bool:
    # Calculates all activations corresponding to `eps_terms` where `zonotope_predictions[label]` achieves minimum.
    min_config = (zonotope_predictions[:, label] < 0).float() * 2 - 1
    min_config[0] = 1.0
    min_config_values = torch.mm(min_config.unsqueeze(0), zonotope_predictions).squeeze()
    print(f"[-] Values @ minimum of `true_label`: {min_config_values}")
    return min_config_values.argmax().item() == label
