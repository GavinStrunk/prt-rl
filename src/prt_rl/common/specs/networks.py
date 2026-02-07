"""
Specification for general networks.

You can use these specs to configure a network, but the intent is to have higher-level policy components that use the network classes directly.
"""
from dataclasses import dataclass
from typing import Tuple, Literal, Optional

@dataclass
class CNNNetworkSpec:
    """
    Specification for a general CNN network.

    Intended usage:
      - Encoder backbone: pooling="avg"/"max", output_dim=None => returns [B, latent_dim]
      - Feature map backbone: pooling=None => returns [B, C, H', W']
      - Encoder+head: pooling="avg"/"max", output_dim=... => returns [B, output_dim]
    """
    kind: Literal["cnn"] = "cnn"

    in_channels: int = 3
    input_hw: Optional[Tuple[int, int]] = None  # set if you want to reason about spatial sizes

    # Conv stack
    channels: Tuple[int, ...] = (32, 64, 64)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (2, 2, 1)
    paddings: Tuple[int, ...] = (1, 1, 1)

    activation: Literal["relu", "tanh"] = "relu"
    norm: Optional[Literal["batch", "layer", "group"]] = None
    group_norm_groups: int = 8
    dropout: float = 0.0

    # Readout
    pooling: Optional[Literal["avg", "max"]] = "avg"
    output_dim: Optional[int] = None
    output_activation: Optional[Literal["relu", "tanh"]] = None

@dataclass
class MLPNetworkSpec:
    """
    Specification for a general MLP network.

    Intended usage:
      - Policy backbone: set output_dim=None and use latent_dim as feature size
      - Policy/critic head: set output_dim to action/value dimension
    """
    kind: Literal["mlp"] = "mlp"
    hidden_sizes: Tuple[int, ...] = (128, 128)
    activation: Literal["relu", "tanh"] = "relu"
    output_dim: Optional[int] = None
    output_activation: Optional[Literal["relu", "tanh"]] = None