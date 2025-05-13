from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class NatureCNN(nn.Module):
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

    The standard MLP architecture is as follows:
        - Linear(64*7*7, feature_dim)
        - ReLU
        - Linear(feature_dim, action_len)

    The dueling architecture is as follows:
        - Advantage stream:
            - Linear(64*7*7, feature_dim)
            - ReLU
            - Linear(feature_dim, action_len) (advantage)
        - Value stream:
            - Linear(64*7*7, feature_dim)
            - ReLU
            - Linear(feature_dim, 1) (value)
        - Combine advantage and value to get Q-values

    Args:
        state_shape (tuple): Shape of the input state tensor (channels, height, width)
        action_len (int): Number of output actions
        feature_dim (int): Number of features in the hidden layer
        dueling (bool): If True, use dueling architecture. Default is False.
    """
    def __init__(self,
                 state_shape: tuple,
                 action_len: int = 4,
                 feature_dim: int = 512,
                 dueling: bool = False,
                 ) -> None:
        super(NatureCNN, self).__init__()
        self.dueling = dueling

        if len(state_shape) != 3:
            raise ValueError("state_shape must be a tuple of (channels, height, width)")

        num_channels = state_shape[0]

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the size of the feature map after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_shape)
            conv_out = self._forward_conv(dummy_input)
            conv_out_dim = conv_out.view(1, -1).size(1)

        if not self.dueling:
            self.fc1 = nn.Linear(conv_out_dim, feature_dim)
            self.fc2 = nn.Linear(feature_dim, action_len)
        else:
            self.fc1_adv = nn.Linear(conv_out_dim, feature_dim)
            self.fc2_adv = nn.Linear(feature_dim, action_len)
            self.fc1_val = nn.Linear(conv_out_dim, feature_dim)
            self.fc2_val = nn.Linear(feature_dim, 1)    

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dtype == torch.uint8:
            state = state.float() / 255.0

        x = self._forward_conv(state)
        x = x.view(x.size(0), -1)

        if not self.dueling:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            adv = F.relu(self.fc1_adv(x))
            adv = self.fc2_adv(adv)
            val = F.relu(self.fc1_val(x))
            val = self.fc2_val(val)
            x = val + (adv - adv.mean(dim=1, keepdim=True))
        return x
    
