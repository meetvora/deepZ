import copy
import torch
import torch.nn as nn

INPUT_SIZE = 28


class Model(nn.Module):
    """Creates a copy of provided model with noise params as new images, along batch axis."""

    def __init__(self, model: nn.Module, k: int, eps: float):
        super().__init__()
        layers = [self.modify(layer) for layer in model.layers]
        self.net = nn.Sequential(*layers)
        self.setNoiseParams(k, eps)

    def setNoiseParams(self, k: int, eps: float):
        self.noise_params = torch.FloatTensor(k, 1, INPUT_SIZE,
                                              INPUT_SIZE).uniform_(0, 1)
        self.noise_params = (self.noise_params * eps) / self.noise_params.sum(
            dim=0)[0].unsqueeze(0).unsqueeze(0)
        self.noise_params = self.noise_params * (
            2 * torch.randint(0, 2, self.noise_params.shape) - 1).float()
        assert torch.all(torch.abs(self.noise_params).sum(dim=0) == eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zonotope_input = torch.cat([x, self.noise_params], dim=0)
        return self.net(zonotope_input)

    def modify(self, layer: nn.Module) -> nn.Module:
        layer_name = layer.__class__.__name__
        modified_layers = {
            'Normalization': Normalization,
            'Linear': Linear,
            'ReLU': ReLU
        }

        if layer_name not in modified_layers:
            return copy.deepcopy(layer)

        return modified_layers[layer_name](layer)

    def updateConvexApprox(self):
        #TODO. Ideas: Update slope and/or k
        pass


class Linear(nn.Module):
    """ Modified Linear layer such that bias is added only to the first sample in batch"""

    def __init__(self, layer: torch.Tensor):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rectified_bias = self.layer.bias.data.unsqueeze(0).repeat(len(x), 1)
        rectified_bias[0] *= 0
        return self.layer(x) + rectified_bias


class Normalization(nn.Module):
    """Modified Normalization layer such that bias is added only to the first sample in batch"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[0] = (x[0] - 0.1307) / 0.3081
        return x


class ReLU(nn.Module):
    """Modified ReLU that tries to approximate the Zonotope region.
    Adds a new sample to batch sample, if new noise parameter is added to system."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.slope = 0
        self.intercept = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise_magnitude = torch.abs(x[1:]).sum(dim=0)
        lower_bound, upper_bound = (x[0] - noise_magnitude,
                                    x[0] + noise_magnitude)

        neg_mask = (upper_bound <= 0)
        pos_mask = (lower_bound >= 0)
        x *= (~neg_mask
             ).float()  # Setting all values 0 if upper_bound is non-positive.
        mask = ~(neg_mask + pos_mask)  # Mask is 1 where crossing takes place.

        return self.convexApprox(x, upper_bound, lower_bound,
                                 mask) if torch.any(mask) else x

    def convexApprox(self, x: torch.Tensor, upper_bound: torch.Tensor,
                     lower_bound: torch.Tensor, mask: torch.Tensor):
        #TODO: Improve slope calculation algorithm.

        # Currently finds least area zonotope.
        self.slope = upper_bound / (upper_bound - lower_bound)
        self.intercept = -(self.slope * lower_bound) / 2
        x[0] = self.slope * x[0] + self.intercept
        new_noise_param = (torch.ones_like(x[0]) *
                           self.intercept) * mask.float()
        x = torch.cat([x, new_noise_param.unsqueeze(0)], dim=0)
        return x
