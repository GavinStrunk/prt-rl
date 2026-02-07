from torch import nn
from typing import Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.components.encoders.interface import EncoderInterface
from prt_rl.common.components.encoders.registry import register_encoder
from prt_rl.common.specs.encoders import IdentityEncoderSpec


class IdentityEncoder(EncoderInterface):
    def __init__(self, input_dim: Tuple[int, ...]):
        super().__init__()
        # @todo flatten if the dimension is more than 2D
        self.out_dim = input_dim[-1]

    @property
    def latent_dim(self) -> int:
        return self.out_dim
    
    def forward(self, x):  # x: obs
        return x
    
@register_encoder("identity")
def build_identity_encoder(env_params: EnvParams, spec: IdentityEncoderSpec) -> nn.Module:
    return IdentityEncoder(env_params.observation_shape)