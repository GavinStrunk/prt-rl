from abc import ABC, abstractmethod
import torch
import torch.distributions as tdist
from typing import Tuple, Union
from prt_rl.env.interface import EnvParams

class Distribution(ABC):
    @staticmethod
    @abstractmethod
    def get_action_dim(env_params: EnvParams) -> int:
        """
        Returns the number of parameters per action to define the distribution.

        Returns:
            int: The number of parameters required for each action in the distribution.
        """
        raise NotImplementedError("This method should be implemented by subclasses to return the number of parameters per action.")
    
    @staticmethod
    @abstractmethod
    def last_network_layer(feature_dim: int, action_dim: int, **kwargs) -> Union[torch.nn.Module, Tuple[torch.nn.Module, torch.nn.Parameter]]:
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
    def get_action_dim(env_params: EnvParams) -> int:
        """
        Returns the number of parameters per action to define the distribution.
        For Categorical, this is the number of actions in the environment.
        Args:
            env_params (EnvParams): The environment parameters containing the action space.
        Returns:
            int: The number of actions in the environment.
        """
        if env_params.action_continuous:
            raise ValueError("Categorical distribution is not suitable for continuous action spaces.")
        return env_params.action_max - env_params.action_min + 1

    @staticmethod
    def last_network_layer(feature_dim: int, action_dim: int) -> torch.nn.Module:
        """
        Returns the last layer of the network that produces the parameters for this distribution.
        For Categorical, this is a linear layer with num_actions outputs.
        Args:
            num_actions (int): The number of actions in the environment.
        """
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=feature_dim, out_features=action_dim),
            torch.nn.Softmax(dim=-1)
        )

class Normal(Distribution, tdist.Normal):
    """
    Multivariate Normal or Diagonal Gaussian distribution.
    This distribution is state independent parameterized by a mean and a log standard deviation (or scale).
    
    .. math::
        a \sim N(\mu(s), exp(log(\sigma(s)^2) I)
    
    Args:
        probs (torch.Tensor): A tensor of means with shape (B, num_actions) 
        parameters (torch.nn.Parameter): A tensor of log standard deviations with shape (num_actions, ).
    """
    def __init__(self,
                 probs: torch.Tensor,
                 parameters: torch.nn.Parameter
                 ) -> None:
        if len(probs.shape) != 2:
            raise ValueError("Normal distribution requires probs to have shape (B, num_actions)")
        
        super().__init__(probs, torch.exp(parameters))

    @staticmethod
    def get_action_dim(env_params: EnvParams) -> int:
        """
        Returns the number of parameters per action to define the distribution.
        For Normal, this is 2 * num_actions (mean and log standard deviation).
        Args:
            env_params (EnvParams): The environment parameters containing the action space.
        Returns:
            int: The number of parameters required for each action in the distribution.
        """
        if not env_params.action_continuous:
            raise ValueError("Normal distribution is only suitable for continuous action spaces.")
        return env_params.action_len

    @staticmethod
    def last_network_layer(feature_dim: int, action_dim: int, log_std_init: float = 0.0) -> Tuple[torch.nn.Module, torch.nn.Parameter]:
        """
        Returns the last layer of the network that produces the parameters for this distribution.
        For Normal, this is a linear layer with 2 * num_actions outputs (mean and std).
        Args:
            num_actions (int): The number of actions in the environment.
        Returns:
            torch.nn.Module: The last layer of the network that produces the parameters for this distribution.
            torch.nn.Parameter: A parameter for the log standard deviation, initialized to zero.
        """
        log_std = torch.nn.Parameter(torch.ones(action_dim) * log_std_init, requires_grad=True)
        
        return torch.nn.Linear(in_features=feature_dim, out_features=action_dim), log_std