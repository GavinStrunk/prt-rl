from abc import ABC, abstractmethod
import math
import torch
import torch.distributions as tdist
from typing import Tuple, Union, Optional
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
    
    @abstractmethod
    def deterministic_action(self) -> torch.Tensor:
        """
        Returns a deterministic action based on the distribution parameters.
        This is used for evaluation or inference where we want to select the most probable action.
        
        Returns:
            torch.Tensor: A tensor representing the deterministic action.
        """
        raise NotImplementedError("This method should be implemented by subclasses to return a deterministic action.")
        

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
    
    def deterministic_action(self) -> torch.Tensor:
        """
        Returns a deterministic action based on the distribution parameters.
        For Categorical, this is simply the index of the maximum probability.
        
        Returns:
            torch.Tensor: A tensor representing the deterministic action.
        """
        return self.probs.argmax(dim=-1)
    
    def sample(self) -> torch.Tensor:
        """
        Samples an action from the Categorical distribution.
        The output is a tensor with shape (batch_size, 1) where each element is the sampled action index.
        
        Returns:
            torch.Tensor: A tensor containing the sampled actions with shape (batch_size, 1).
        """
        return super().sample().unsqueeze(-1)

class Normal(Distribution, tdist.Normal):
    """
    Multivariate Normal or Diagonal Gaussian distribution.

    This distribution is state independent parameterized by a mean and a log standard deviation (or scale). This distribution expects a neural network to output the mean for each action and treats the log standard deviations as free parameters.
    
    .. math::
        a \sim N(\mu(s), exp(log(\sigma(s)^2) I)
    
    Args:
        mu (torch.Tensor): A tensor of means with shape (B, num_actions) 
        log_std (torch.nn.Parameter): A tensor of log standard deviations with shape (num_actions, ).
    """
    def __init__(self,
                 mu: torch.Tensor,
                 log_std: torch.nn.Parameter
                 ) -> None:
        if len(mu.shape) != 2:
            raise ValueError("Normal distribution requires probs to have shape (B, num_actions)")
        
        # Pytorch Normal expects the scale (std) instead of variance
        super().__init__(mu, torch.exp(log_std))

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
        For Normal, this is a linear layer with num_actions outputs (mean). The log standard deviation is treated as a free parameter and is initialized to zero by default.
        Args:
            num_actions (int): The number of actions in the environment.
        Returns:
            torch.nn.Module: The last layer of the network that produces the parameters for this distribution.
            torch.nn.Parameter: A parameter for the log standard deviation, initialized to zero.
        """
        log_std = torch.nn.Parameter(torch.ones(action_dim) * log_std_init, requires_grad=True)
        
        return torch.nn.Linear(in_features=feature_dim, out_features=action_dim), log_std
    
    def deterministic_action(self) -> torch.Tensor:
        """
        Returns a deterministic action based on the distribution parameters.
        For Normal, this is simply the mean of the distribution.
        
        Returns:
            torch.Tensor: A tensor representing the deterministic action.
        """
        return self.mean
    
    def sample(self):
        return super().rsample()
    
class TanhGaussian(Normal):
    """
    Tanh-transformed Gaussian to constrain the actions to the range [-1, 1]. 
    
    This distribution is typically used in SAC and expects a neural network to output the mean for each action and treats the diagonal log standard deviation as free parameters. This means the standard deviation is not state dependent.

    .. math::
        a = tanh(u), where u ~ \mathcal{N}(\mu, diag(\sigma))

    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution with shape (B, action_dim).
        log_std (torch.Tensor): Log standard deviation of the Gaussian distribution with shape (action_dim, ).
    """
    def __init__(
        self,
        mu: torch.Tensor,         
        log_std: torch.Tensor,  
    ) -> None:
        super().__init__(mu, log_std)

    @staticmethod
    def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Inverse of the tanh function.
        
        .. math::
            atanh(x) = 0.5 * \log((1 + x) / (1 - x))
        
        Args:
            x (torch.Tensor): Input tensor.
            eps (float): Small value to avoid numerical issues at the boundaries.
        Returns:
            torch.Tensor: The inverse tanh of the input tensor."""
        # Clamp to open interval (-1, 1) for numerical stability
        x = x.clamp(min=-1 + eps, max=1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))        

    def deterministic_action(self) -> torch.Tensor:
        """
        Push the mean through tanh.
        """
        return torch.tanh(self.mean)
    
    def log_prob(self, value):
        # Compute u from tanh inverse of action value
        u = self._atanh(value)

        # Compute the Base Normal log prob
        log_prob = super().log_prob(u).sum(-1, keepdim=True)

        # Compute the log det jacobian of the tanh transform
        log2 = torch.log(torch.tensor(2.0, device=u.device, dtype=u.dtype))
        log_det = 2 * (log2 - u - torch.nn.functional.softplus(-2 * u))
        log_det = log_det.sum(-1, keepdim=True)  # sum over action
        return log_prob - log_det
    
    def entropy(self):
        """
        No analytical form for entropy of transformed distribution. Entropy can be estimated using -log_prob.mean()
        """
        return None
    
    def sample(self) -> torch.Tensor:
        # Non-reparameterized sample and then squash with tanh
        return torch.tanh(super().rsample())

