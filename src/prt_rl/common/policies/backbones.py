from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union, Literal

def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

@dataclass
class MLPBackboneSpec:
    """
    Specification for an MLP backbone.

    Args:
        network_arch (List[int]): List of integers specifying the number of units in each hidden layer.
        hidden_activation (torch.nn.Module): Activation function to use for hidden layers.
    """
    kind: Literal["mlp"] = "mlp"
    hidden_sizes: Tuple[int, ...] = (128, 128)
    activation: Literal["relu", "tanh"] = "relu"



class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], activation: str):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), _activation(activation)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.latent_dim = prev

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

BackboneSpec = Union[MLPBackboneSpec]

def build_backbone(latent_dim: int, spec: BackboneSpec) -> MLPBackbone:
    if isinstance(spec, MLPBackboneSpec):
        return MLPBackbone(latent_dim, spec.hidden_sizes, spec.activation)
    raise TypeError(f"Unsupported backbone spec: {type(spec)}")