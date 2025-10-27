from abc import ABC, abstractmethod
import torch
from typing import Tuple
from prt_rl.env.interface import EnvParams


class BasePolicy(torch.nn.Module, ABC):
    """
    Base class for implementing policies.

    Args:
        env_params (EnvParams): Environment parameters.
    """
    def __init__(self,
                 env_params: EnvParams,
                 ) -> None:
        super().__init__()
        self.env_params = env_params

    def __call__(self,
                   state: torch.Tensor,
                   deterministic: bool = False
                   ) -> torch.Tensor:
        return self.forward(state, deterministic=deterministic)

    def forward(self,
                   state: torch.Tensor,
                   deterministic: bool = False
                   ) -> torch.Tensor:
        """
        Chooses an action based on the current state. 

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.

        Returns:
            torch.Tensor: Tensor with the chosen action.
        """
        action, _, _ = self.predict(state, deterministic=deterministic)
        return action
    
    @abstractmethod
    def predict(self,
                state: torch.Tensor,
                deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chooses an action based on the current state and returns the action, value estimate, and log probability.
        
        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the chosen action, value estimate, and action log probability.
                - action (torch.Tensor): Tensor with the chosen action. Shape (B, action_dim)
                - value_estimate (torch.Tensor): Tensor with the estimated value of the state. Shape (B, 1)
                - log_prob (torch.Tensor): Tensor with the log probability of the chosen action. Shape (B, 1)
        """
        raise NotImplementedError