from torch import nn, Tensor
from typing import Optional, Tuple
from prt_rl.common.utils import to_activation


class MLPNetwork(nn.Module):
    """
    General-purpose MLP for use in policies/critics/heads.

    Notes:
      - This class builds only the hidden stack. Use `output_dim` + `output_activation`
        to optionally add an output layer.
      - `latent_dim` is the final hidden size if no output layer is created; otherwise
        it is the output_dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...],
        activation: str,
        output_dim: Optional[int] = None,
        output_activation: Optional[str] = None,
    ):
        super().__init__()
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

        self.net = nn.Sequential(*layers)
        self.latent_dim = prev

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)