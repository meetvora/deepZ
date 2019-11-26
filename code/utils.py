import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def verify(zonotope_predictions: torch.Tensor, true_label: int) -> bool:
    """
    Idea: a) Calculate values for other neurons at the point where `zonotope_predictions[true_label]` achieves it's 
          minimum value. Check if value @ `true_label` is highest.
          b) The, for all other labels, find the point where `zonotope_predictions[label]` achieves it's maximum. Check
          if value @ `true_label` is highest.

    TODO: a) Input values can be b/w [0, 1]. Certain `eps_term` in Zonotope might change their upper or lower bound.
          b) `condition` in `isExtremumValid` checks for negative & positive values. Extend to non-positive & 
             non-negative
          c) Other points to check so that verification is sound and complete.
    """

    def isExtremumValid(minima: bool, label: int) -> torch.Tensor:
        condition = (zonotope_predictions[:, label] < 0) if minima else (zonotope_predictions[:, label] > 0)
        eps_config = condition.float() * 2 - 1
        eps_config[0] = 1.0
        eps_config_values = torch.mm(eps_config[None, :], zonotope_predictions).squeeze()

        logger.debug(f"Values @ {'minimum' if minima else 'maximum'} of `label: {label}`: {eps_config_values}")

        return eps_config_values.argmax().item() == true_label

    if not isExtremumValid(minima=True, label=true_label):
        return False

    for label in range(10):
        if not isExtremumValid(minima=False, label=label):
            return False

    return True
