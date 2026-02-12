from torch import nn, Tensor
from typing import Optional, Tuple
from prt_rl.common.utils import to_activation


def build_mlp(
    input_dim: int,
    hidden_sizes: Tuple[int, ...],
    activation: str,
    output_dim: Optional[int] = None,
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    layers = []
    prev = input_dim

    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), to_activation(activation)]
        prev = h

    if output_dim is not None:
        layers += [nn.Linear(prev, output_dim)]
        prev = output_dim
        if output_activation is not None:
            layers += [to_activation(output_activation)]

    return nn.Sequential(*layers)
