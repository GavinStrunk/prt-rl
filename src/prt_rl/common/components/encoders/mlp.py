from torch import nn
from typing import Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.components.encoders.interface import EncoderInterface
from prt_rl.common.components.encoders.registry import register_encoder
from prt_rl.common.specs.encoders import MLPEncoderSpec

class MLPEncoder(EncoderInterface):
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

@register_encoder("mlp_encoder")
def build_mlp_encoder(env_params: EnvParams, spec: MLPEncoderSpec) -> nn.Module:
    if len(env_params.observation_shape) != 1:
        raise ValueError("MLPEncoder expects 1D obs.")
    return MLPEncoder(env_params.observation_shape[0], spec.hidden_sizes, spec.activation)
