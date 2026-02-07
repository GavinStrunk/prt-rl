from torch import nn
from prt_rl.env.interface import EnvParams
from prt_rl.common.components.encoders.interface import EncoderInterface
from prt_rl.common.components.encoders.registry import register_encoder
from prt_rl.common.specs.encoders import NatureCNNEncoderSpec
from torch import Tensor
import torch

class NatureCNNEncoder(EncoderInterface):
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