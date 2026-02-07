from dataclasses import dataclass
from typing import Tuple, Literal, Union

@dataclass
class MLPBackboneSpec:
    """
    Specification for an MLP backbone.

    Args:
        network_arch (List[int]): List of integers specifying the number of units in each hidden layer.
        hidden_activation (torch.nn.Module): Activation function to use for hidden layers.
    """
    kind: Literal["mlp"] = "mlp"
    hidden_sizes: Tuple[int, ...] = (128, 128)
    activation: Literal["relu", "tanh"] = "relu"

BackboneSpec = Union[
    MLPBackboneSpec,
]