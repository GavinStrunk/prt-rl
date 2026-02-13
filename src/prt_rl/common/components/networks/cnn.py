import torch
from torch import nn, Tensor
from typing import Optional, Tuple, Literal
from prt_rl.common.utils import to_activation

def _maybe_norm(kind: Optional[str], num_channels: int) -> nn.Module:
    if kind is None:
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm2d(num_channels)
    if kind == "layer":
        # LayerNorm over channel+spatial is awkward for conv; GroupNorm(1, C) is a common proxy.
        return nn.GroupNorm(1, num_channels)
    if kind == "group":
        # Default groups handled in builder; this placeholder is replaced if needed.
        return nn.Identity()
    raise ValueError(f"Unknown norm kind: {kind}")

class CNNNetwork(nn.Module):
    """
    General-purpose CNN network for use in encoders/backbones.

    Builds a conv stack and optionally:
      - applies global pooling (avg/max) to get [B, C]
      - applies a linear output head to get [B, output_dim]

    `latent_dim` refers to:
      - output_dim if output head is present
      - otherwise the channel dimension after pooling (if pooled)
      - otherwise the final conv channel count (if not pooled)
    """

    def __init__(
        self,
        in_channels: int,
        input_hw: Optional[Tuple[int, int]],
        channels: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...],
        strides: Tuple[int, ...],
        paddings: Tuple[int, ...],
        activation: str,
        norm: Optional[str] = None,
        group_norm_groups: int = 8,
        dropout: float = 0.0,
        pooling: Optional[Literal["avg", "max"]] = "avg",
        output_dim: Optional[int] = None,
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        if not (len(channels) == len(kernel_sizes) == len(strides) == len(paddings)):
            raise ValueError("channels/kernel_sizes/strides/paddings must have same length")

        layers: list[nn.Module] = []
        prev_c = in_channels

        for i, out_c in enumerate(channels):
            layers.append(nn.Conv2d(prev_c, out_c, kernel_sizes[i], stride=strides[i], padding=paddings[i]))

            if norm == "group":
                layers.append(nn.GroupNorm(group_norm_groups, out_c))
            else:
                layers.append(_maybe_norm(norm, out_c))

            layers.append(to_activation(activation))

            if dropout and dropout > 0.0:
                layers.append(nn.Dropout2d(p=dropout))

            prev_c = out_c

        self.conv = nn.Sequential(*layers)

        if pooling is None:
            self.pool = nn.Identity()
            pooled_dim = prev_c
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            pooled_dim = prev_c
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            pooled_dim = prev_c
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        head_layers: list[nn.Module] = []
        if output_dim is not None:
            head_layers.append(nn.Flatten(start_dim=1))
            head_layers.append(nn.Linear(pooled_dim, output_dim))
            if output_activation is not None:
                head_layers.append(to_activation(output_activation))
            self.head = nn.Sequential(*head_layers)
            self.latent_dim = output_dim
        else:
            # If we pooled, output will be [B, C, 1, 1] unless we flatten in forward
            self.head = nn.Identity()
            self.latent_dim = pooled_dim

        # Optional: store for shape reasoning (no implementation guarantee)
        self.input_hw = input_hw
        self.pooling = pooling
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            If output_dim is set: [B, output_dim]
            Else if pooling is set: [B, C] (flattened)
            Else: [B, C, H', W']
        """
        y = self.conv(x)
        y = self.pool(y)

        # Normalize outputs to common shapes
        if self.output_dim is not None:
            return self.head(y)

        if self.pooling is not None:
            return torch.flatten(y, start_dim=1)  # [B, C]

        return y  # [B, C, H', W']