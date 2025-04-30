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

class NatureCNN(nn.Sequential):
    """
    Convolutional Neural Network as described in the Nature paper

    The Nature CNN expects a 3D input image tensor with shape (channels, height, width) and values scaled to [0, 1]. The output is a tensor with shape (batch_size, action_len).
    The CNN architecture is as follows:
        - Conv2d(32, kernel_size=8, stride=4)
        - ReLU
        - Conv2d(64, kernel_size=4, stride=2)
        - ReLU
        - Conv2d(64, kernel_size=3, stride=1)
        - ReLU
        - Flatten
        - Linear(64*7*7, feature_dim)
        - ReLU
        - Linear(feature_dim, action_len)

    Args:
        state_shape (tuple): Shape of the input state tensor (channels, height, width)
        action_len (int): Number of output actions
        feature_dim (int): Number of features in the hidden layer
    """
    def __init__(self,
                 state_shape: tuple,
                 action_len: int = 4,
                 feature_dim: int = 512
                 ) -> None:
        if len(state_shape) != 3:
            raise ValueError("state_shape must be a tuple of (channels, height, width)")
        
        # Get the number of channels from the state shape
        num_channels = state_shape[0]
        layers = [
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_len),
        ]
        super(NatureCNN, self).__init__(*layers)

    def forward(self,
                state: torch.Tensor
                ) -> torch.Tensor:
        if state.dtype == torch.uint8:
            # Convert uint8 to float32 and scale to [0, 1]
            state = state.float() / 255.0
            
        return super().forward(state)