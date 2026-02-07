from typing import Callable, Dict
from torch import nn
from prt_rl.env.interface import EnvParams
from prt_rl.common.specs.encoders import EncoderSpec

EncoderBuilder = Callable[[EnvParams, EncoderSpec], nn.Module]
ENCODER_REGISTRY: Dict[str, EncoderBuilder] = {}

def register_encoder(kind: str):
    def _wrap(fn: EncoderBuilder) -> EncoderBuilder:
        ENCODER_REGISTRY[kind] = fn
        return fn
    return _wrap

def build_encoder(env_params: EnvParams, spec: EncoderSpec) -> nn.Module:
    try:
        builder = ENCODER_REGISTRY[spec.kind]
    except KeyError:
        raise TypeError(f"Unsupported encoder spec kind: {spec.kind}")
    return builder(env_params, spec)
