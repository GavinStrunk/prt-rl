from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-layer perceptron network

    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        network_arch (List[int], optional): Hidden layer sizes. Defaults to [64, 64].
        hidden_activation (nn.Module): Activation function for hidden layers.
        final_activation (Optional[nn.Module]): Optional activation after final layer.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: Optional[int] = None,
                 network_arch: Optional[List[int]] = [64, 64],
                 hidden_activation: nn.Module = nn.ReLU(),
                 final_activation: Optional[nn.Module] = None
                 ) -> None:
        super().__init__()
        if output_dim is None and network_arch is None:
            raise ValueError("Either output_dim or network_arch must be provided.")
        
        self.layers = nn.ModuleList()

        if network_arch is None:
            dims = [input_dim, output_dim]
        else:
            dims = [input_dim] + network_arch

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.layers.append(hidden_activation)

        if output_dim is not None:
            self.layers.append(nn.Linear(dims[-1], output_dim))
            self.final_activation = final_activation
        else:
            self.final_activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

class DuelingMLP(nn.Module):
    """
    Dueling Multi-layer Perceptron network
    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        network_arch (Optional[List[int]], optional): Hidden layer sizes. Defaults to None.
        hidden_activation (nn.Module): Activation function for hidden layers.
        final_activation (Optional[nn.Module]): Optional activation after final layer.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 network_arch: Optional[List[int]] = None,
                 hidden_activation: nn.Module = nn.ReLU(),
                 final_activation: Optional[nn.Module] = None
                 ) -> None:
        super().__init__()

        if network_arch is None:
            dims = [input_dim, output_dim]
        else:
            dims = [input_dim] + network_arch + [output_dim]

        self.advantage_layers = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.advantage_layers.append(nn.Linear(dims[i], dims[i + 1]))

            if hidden_activation is not None:
                self.advantage_layers.append(hidden_activation)

        self.advantage_layers.append(nn.Linear(dims[-2], dims[-1]))
        if final_activation is not None:
            self.advantage_layers.append(final_activation)

        self.value_layers = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.value_layers.append(nn.Linear(dims[i], dims[i + 1]))

            if hidden_activation is not None:
                self.value_layers.append(hidden_activation)

        self.value_layers.append(nn.Linear(dims[-2], 1))
        if final_activation is not None:
            self.value_layers.append(final_activation)

    def forward(self, 
                state: torch.Tensor
                ) -> torch.Tensor:
        
        adv = state
        for layer in self.advantage_layers:
            adv = layer(adv)

        val = state
        for layer in self.value_layers:
            val = layer(val)

        # Combine advantage and value to get Q-values
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values


class BaseEncoder(nn.Module):
    def __init__(self, 
                 features_dim: int
                 ) -> None:
        super(BaseEncoder, self).__init__()
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim
    
class NatureCNNEncoder(BaseEncoder):
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
        - Linear(output_dim=feature_dim)
        - ReLU
    """
    def __init__(self,
                 input_shape: tuple,
                 features_dim: int = 512,
                ) -> None:
        super().__init__(features_dim=features_dim)

        if len(input_shape) != 3:
            raise ValueError("state_shape must be a tuple of (channels, height, width)")
        
        num_channels = input_shape[0]

        self.layers = nn.ModuleList([
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ])

        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for layer in self.layers:
                x = layer(x)
            conv_out_dim = x.view(1, -1).size(1)

        self.layers.append(nn.Linear(conv_out_dim, features_dim))
        self.layers.append(nn.ReLU())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dtype == torch.uint8:
            state = state.float() / 255.0

        for layer in self.layers:
            state = layer(state)

        return state

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
        - Linear(output_dim=feature_dim)

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
    
