import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import modLayer, ReLU

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

INPUT_SIZE = 28
NUM_CLASSES = 10


class Model(nn.Module):
    """
    Creates a copy of provided model with `eps_terms` as new images, along batch axis.

    attributes:
    ==========

    `net` (nn.Module): The NN with modified layers (in `layers.py`).
    `eps_term` (torch.FloatTensor): Constant. Independent 'epsilon terms' defined to construct L-inf norm perturbation.
                                    Shape: (768 * 768, 1, 28, 28).
    `true_label` (int): As name suggests.
    `_max_config_values` (torch.Tensor): For each label 'l', we calculate values of final epsilon terms such that 
                                         score[l] obtains its maxima. Shape: (10, 10).
    `_min_config_values` (torch.Tensor): For our true label, we calculate values of final epsilon terms such that
                                         score[true_label] obtains it's minima. Shape: (10, )
    """

    def __init__(self, model: nn.Module, eps: float, true_label: int):
        super().__init__()
        layers = [modLayer(layer) for layer in model.layers]
        self.net = nn.Sequential(*layers)
        self.eps_terms = torch.diag(torch.ones(INPUT_SIZE * INPUT_SIZE) * eps)
        self.eps_terms = self.eps_terms.reshape((INPUT_SIZE * INPUT_SIZE, 1, INPUT_SIZE, INPUT_SIZE))
        self.true_label = true_label
        self._max_config_values = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        box_input = torch.cat([x, self.eps_terms], dim=0)
        return self.net(box_input)

    def updateParams(self):
        # Calculates the gradient of `loss` wrt to ReLU slopes.
        # TODO: Improve training objective.

        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, weight_decay=0)
        loss = torch.mean(torch.clamp(self._min_config_values - self._min_config_values[self.true_label], -1))
        for label in range(NUM_CLASSES):
            loss += torch.mean(
                torch.clamp(self._max_config_values[label] - self._max_config_values[label][self.true_label], -1)
            )

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    def parameters(self) -> torch.Tensor:
        # A generator to allow gradient descent on `slope` of ReLUs.

        for layer in self.net:
            if isinstance(layer, ReLU) and hasattr(layer, "slope"):
                yield layer.slope

    @staticmethod
    def getExtremum(predictions: torch.Tensor, minima: bool, label: int) -> torch.Tensor:
        # Calculates extremum values as defined by `_max_config_values` and `_min_config_values`

        condition = (predictions[:, label] < 0) if minima else (predictions[:, label] > 0)
        eps_config = condition.float() * 2 - 1
        eps_config[0] = 1.0
        return torch.mm(eps_config[None, :], predictions).squeeze()

    def verify(self, x: torch.Tensor) -> bool:
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

        self._zono_pred = self.forward(x)

        self._min_config_values = self.getExtremum(self._zono_pred, minima=True, label=self.true_label)
        logger.debug(f"Values @ minimum of `true label: {self.true_label}`: {self._min_config_values}")

        if self._min_config_values.argmax().item() != self.true_label:
            return False

        for label in range(NUM_CLASSES):
            self._max_config_values[label] = self.getExtremum(self._zono_pred, minima=False, label=label)
            logger.debug(f"Values @ maximum of `label: {label}`: {self._max_config_values[label]}")
            if self._max_config_values[label].argmax().item() != self.true_label:
                return False

        return True
