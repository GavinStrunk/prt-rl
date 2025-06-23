from abc import ABC, abstractmethod
import torch
import torch.distributions as tdist

class Distribution(ABC):
    @staticmethod
    @abstractmethod
    def parameters_per_action() -> int:
        """
        Returns the number of parameters per action to define the distribution.

        Returns:
            int: The number of parameters required for each action in the distribution.
        """
        raise NotImplementedError("This method should be implemented by subclasses to return the number of parameters per action.")
    
    @staticmethod
    @abstractmethod
    def last_network_layer(feature_dim: int, num_actions: int) -> torch.nn.Module:
        """
        Returns the last layer of the network that produces the parameters for this distribution.
        This is used to determine the output shape of the policy head.
        Args:
            num_actions (int): The number of actions in the environment.
        Returns:
            torch.nn.Module: The last layer of the network that produces the parameters for this distribution.
        """
        raise NotImplementedError("This method should be implemented by subclasses to return the last layer of the network for this distribution.")
        

class Categorical(Distribution, tdist.Categorical):
    def __init__(self,
                 probs: torch.Tensor
                 ) -> None:
        # Probabilities are passed in with shape (# batch, # actions, # params)
        # Categorical only has 1 param and wants the list with shape (# batch, # action probs) so we squeeze the last dimension
        probs = probs.squeeze(-1)
        super().__init__(probs)

    @staticmethod
    def parameters_per_action() -> int:
        return 1
    
    @staticmethod
    def last_network_layer(feature_dim: int, num_actions: int) -> torch.nn.Module:
        """
        Returns the last layer of the network that produces the parameters for this distribution.
        For Categorical, this is a linear layer with num_actions outputs.
        Args:
            num_actions (int): The number of actions in the environment.
        """
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=feature_dim, out_features=num_actions),
            torch.nn.Softmax(dim=-1)
        )

class Normal(Distribution, tdist.Normal):
    def __init__(self,
                 probs: torch.Tensor
                 ) -> None:
        scale = torch.nn.functional.relu(probs[..., 1])
        super().__init__(loc=probs[..., 0], scale=scale)

    @staticmethod
    def parameters_per_action() -> int:
        return 2
    
    @staticmethod
    def last_network_layer(feature_dim: int, num_actions: int) -> torch.nn.Module:
        """
        Returns the last layer of the network that produces the parameters for this distribution.
        For Normal, this is a linear layer with 2 * num_actions outputs (mean and std).
        Args:
            num_actions (int): The number of actions in the environment.
        """
        return torch.nn.Linear(in_features=feature_dim, out_features=2 * num_actions)