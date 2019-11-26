import copy
import torch
import torch.nn as nn

INPUT_SIZE = 28


class Model(nn.Module):
    """
    Creates a copy of provided model with `eps_terms` as new images, along batch axis.
    """

    def __init__(self, model: nn.Module, eps: float):
        super().__init__()
        layers = [self.modify(layer) for layer in model.layers]
        self.net = nn.Sequential(*layers)
        self.eps_terms = torch.diag(torch.ones(INPUT_SIZE * INPUT_SIZE) * eps)
        self.eps_terms = self.eps_terms.reshape((INPUT_SIZE * INPUT_SIZE, 1, INPUT_SIZE, INPUT_SIZE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zonotope_input = torch.cat([x, self.eps_terms], dim=0)
        return self.net(zonotope_input)

    def modify(self, layer: nn.Module) -> nn.Module:
        layer_name = layer.__class__.__name__
        modified_layers = {"Normalization": Normalization, "Linear": Linear, "ReLU": ReLU}

        if layer_name not in modified_layers:
            return copy.deepcopy(layer)

        return modified_layers[layer_name](layer)

    def updateParams(self):
        # TODO. Idea: Update slope & intercept of present ReLU layers.
        for layer in self.net:
            if isinstance(layer, ReLU):
                layer.updateParams()


class Linear(nn.Module):
    """
    Modified Linear layer such that bias is added only to the first sample in batch.
    """

    def __init__(self, layer: torch.Tensor):
        super().__init__()
        self.weight = layer.weight.data.T
        self.bias = layer.bias.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.mm(x, self.weight)
        y[0] += self.bias
        return y


class Normalization(nn.Module):
    """
    Modified Normalization such that mean is subtracted only to the first sample in batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[0] -= 0.1307
        return x * 3.2457  # Inverted division to multiplication.


class ReLU(nn.Module):
    """
    Modified ReLU that tries to approximate the Zonotope region.
    Adds a new sample to batch sample, if new eps term is added to system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.slope = 0
        self.intercept = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise_magnitude = torch.abs(x[1:]).sum(dim=0)
        lower_bound, upper_bound = (x[0] - noise_magnitude, x[0] + noise_magnitude)

        neg_mask = upper_bound <= 0
        pos_mask = lower_bound >= 0
        x *= (~neg_mask).float()  # Setting all values 0 if upper_bound is non-positive.
        mask = ~(neg_mask + pos_mask)  # Mask is 1 where crossing takes place.

        return self._convexApprox(x, upper_bound, lower_bound, mask.float()) if torch.any(mask) else x

    def _convexApprox(
        self, x: torch.Tensor, upper_bound: torch.Tensor, lower_bound: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.slope == 0 and self.intercept == 0:
            self._leastArea(upper_bound, lower_bound)

        y = self._addNewEpsilon(x, mask)

        del x, lower_bound, upper_bound, mask
        return y

    def _addNewEpsilon(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.slope * x * mask + (1 - mask) * x
        x[0] = (x[0] + self.intercept * 0.5) * mask + (1 - mask) * x[0]
        new_eps_term = (torch.ones_like(x[0]) * self.intercept * 0.5) * mask

        y = torch.cat([x, new_eps_term.unsqueeze(0)], dim=0)
        del x, new_eps_term
        return y

    def _leastArea(self, upper_bound: torch.Tensor, lower_bound: torch.Tensor):
        self.slope = upper_bound / (upper_bound - lower_bound)
        self.intercept = -(self.slope * lower_bound)

    def updateParams(self):
        # Change `self.slope` and `self.intercept`
        pass
