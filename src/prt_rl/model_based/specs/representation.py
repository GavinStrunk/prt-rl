from dataclasses import dataclass, field
from typing import Literal, Union

from prt_rl.common.specs.networks import CNNNetworkSpec


@dataclass
class KeypointRepresentationSpec:
    kind: Literal["keypoint"] = "keypoint"
    num_features: int = 32
    num_keypoints: int = 16
    normalize_coords: bool = False
    backbone: CNNNetworkSpec = field(
        default_factory=lambda: CNNNetworkSpec(pooling=None, output_dim=None)
    )


RepresentationSpec = Union[
    KeypointRepresentationSpec,
]
