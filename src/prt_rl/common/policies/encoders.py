
from dataclasses import dataclass
import torch
from torch import nn
from torch import Tensor
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, Literal
from prt_rl.env.interface import EnvParams

@dataclass
class EncoderOptions:
    trainable: bool = True          # if False -> freeze params
    force_eval: bool = False        # optional: keep dropout/bn frozen too

@dataclass
class IdentityEncoderSpec:
    kind: Literal["identity"] = "identity"

class IdentityEncoder(nn.Module):
    def __init__(self, input_dim: Tuple[int, ...]):
        super().__init__()
        # @todo flatten if the dimension is more than 2D
        self.out_dim = input_dim[-1]

    @property
    def latent_dim(self) -> int:
        return self.out_dim
    
    def forward(self, x):  # x: obs
        return x

@dataclass
class MLPEncoderSpec:
    kind: Literal["mlp_encoder"] = "mlp_encoder"
    hidden_sizes: Tuple[int, ...] = (256,)
    activation: Literal["relu", "tanh"] = "relu"

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], activation: str):
        super().__init__()
        layers = []
        prev = input_dim
        act = nn.ReLU() if activation == "relu" else nn.Tanh()
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), act]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    @property
    def latent_dim(self) -> int:
        return self.out_dim

    def forward(self, x):
        return self.net(x)

@dataclass
class NatureCNNEncoderSpec:
    """
    Specification for the Atari "Nature CNN" encoder (Mnih et al., 2015).

    This is the standard convolutional feature extractor used for Atari deep RL:
      - Conv(32, 8x8, stride=4) + ReLU
      - Conv(64, 4x4, stride=2) + ReLU
      - Conv(64, 3x3, stride=1) + ReLU
      - FC(512) + ReLU

    The paper uses an input of shape 84x84x4 (4 stacked, preprocessed frames).
    You can generalize this via `in_channels` and `input_hw`.

    Args:
        in_channels: Number of input channels (typically 4 for Atari frame stack).
        input_hw: (H, W) of the preprocessed frames (typically (84, 84)).
        feature_dim: Output feature dimension (paper uses 512).
        activation: Nonlinearity used after each hidden layer (paper uses ReLU).
    """
    in_channels: int = 4
    input_hw: Tuple[int, int] = (84, 84)
    feature_dim: int = 512
    activation: str = "relu"


class NatureCNNEncoder(nn.Module):
    """
    Atari Nature CNN encoder (Mnih et al., 2015).

    Maps image observations (B, C, H, W) to a latent feature vector (B, feature_dim).

    Notes:
        - This module assumes inputs are already preprocessed (e.g., resized 84x84,
          stacked frames, dtype/normalization handled upstream).
        - The original paper used 84x84x4 inputs after preprocessing and frame stacking.
    """

    def __init__(self, spec: NatureCNNEncoderSpec) -> None:
        super().__init__()
        self.spec = spec

        act = self._make_activation(spec.activation)

        self.conv = nn.Sequential(
            nn.Conv2d(spec.in_channels, 32, kernel_size=8, stride=4),
            act,
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            act,
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            act,
        )

        # Compute conv output size once (robust to non-84x84 if you want to reuse)
        conv_out_dim = self._infer_conv_out_dim(
            in_channels=spec.in_channels,
            input_hw=spec.input_hw,
        )

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, spec.feature_dim),
            act,
        )

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unknown activation: {name}")

    def _infer_conv_out_dim(self, in_channels: int, input_hw: Tuple[int, int]) -> int:
        h, w = input_hw
        with torch.no_grad():
            x = torch.zeros(1, in_channels, h, w)
            y = self.conv(x)
            return int(y.flatten(1).shape[1])

    @property
    def latent_dim(self) -> int:
        """Latent feature dimension produced by the encoder."""
        return self.spec.feature_dim

    def forward(self, obs: Tensor) -> Tensor:
        """
        Args:
            obs: Observation tensor of shape (B, C, H, W).

        Returns:
            Latent feature tensor of shape (B, feature_dim).
        """
        z = self.conv(obs)
        z = z.flatten(start_dim=1)
        return self.fc(z)

EncoderSpec = Union[IdentityEncoderSpec, MLPEncoderSpec, NatureCNNEncoderSpec]

def build_encoder(env_params: EnvParams, spec: EncoderSpec, options: EncoderOptions) -> nn.Module:
    if isinstance(spec, IdentityEncoderSpec):
        return IdentityEncoder(env_params.observation_shape)
    if isinstance(spec, MLPEncoderSpec):
        if len(env_params.observation_shape) != 1:
            raise ValueError("MLPEncoder expects 1D obs.")
        return MLPEncoder(env_params.observation_shape[0], spec.hidden_sizes, spec.activation)
    raise TypeError(f"Unsupported encoder spec: {type(spec)}")
