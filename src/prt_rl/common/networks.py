from typing import Optional, List
import torch
import torch.nn as nn


class MLP(nn.Sequential):
    """
    Multi-layer perceptron network

    Args:
        state_dim (int): Number of input states
        action_dim (int): Number of output actions
        network_arch: List of hidden nodes
        hidden_activation: Activation function applied to hidden nodes
        final_activation: Activation function applied to output nodes
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 network_arch: List[int] = None,
                 hidden_activation: nn.Module = nn.ReLU(),
                 final_activation: Optional[nn.Module] = None
                 ) -> None:
        # Default architecture is state:64 -> 64:64 -> 64:action
        if network_arch is None:
            network_arch = [64, 64]

        dimensions = [state_dim] + network_arch + [action_dim]

        # Create layers
        layers = []
        for i in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(hidden_activation)

        # Create the final linear layer
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))

        # Add activation after final linear layer if specified
        if final_activation is not None:
            layers.append(final_activation)

        super(MLP, self).__init__(*layers)

    def forward(self,
                state: torch.Tensor
                ) -> torch.Tensor:
        return super().forward(state)

    def init_args(self) -> dict:
        """
        Returns a dictionary of arguments passed to __init__

        Returns:
            dict: Initialization arguments
        """
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'network_arch': self.network_arch,
            'hidden_activation': self.hidden_activation,
            'final_activation': self.final_activation
        }
