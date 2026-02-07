from dataclasses import dataclass
from typing import Literal, Tuple, Union

@dataclass
class EncoderOptions:
    trainable: bool = True          # if False -> freeze params
    force_eval: bool = False        # optional: keep dropout/bn frozen too

@dataclass
class IdentityEncoderSpec:
    kind: Literal["identity"] = "identity"

@dataclass
class MLPEncoderSpec:
    kind: Literal["mlp_encoder"] = "mlp_encoder"
    hidden_sizes: Tuple[int, ...] = (256,)
    activation: Literal["relu", "tanh"] = "relu"

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
    kind: Literal["nature_cnn"] = "nature_cnn"
    in_channels: int = 4
    input_hw: Tuple[int, int] = (84, 84)
    feature_dim: int = 512
    activation: Literal["relu", "tanh", "gelu"] = "relu"

EncoderSpec = Union[
    IdentityEncoderSpec,
    MLPEncoderSpec,
    NatureCNNEncoderSpec,
]