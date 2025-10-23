import torch
from typing import Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies.base import BasePolicy
from prt_rl.common.decision_functions import DecisionFunction, EpsilonGreedy
from prt_rl.common.qtable import QTable

class QTablePolicy(BasePolicy):
    """
    QTablePolicy is a policy that uses a Q-table to select actions for discrete action spaces. It can optionally use a decision function to select actions based on the Q-values.

    The architecture of the policy is as follows:
        - Q-Table: Stores the Q-values for each state-action pair.
        - Decision Function: Selects actions based on the Q-values.
    
    """
    def __init__(self, 
                 env_params: EnvParams,
                 decision_function: DecisionFunction | None = None,
                 ) -> None:
        super().__init__(env_params)

        if env_params.action_continuous:
            raise ValueError("QTablePolicy does not support continuous action spaces. Use a different policy class.")
        
        # Get action dimension
        action_dim = env_params.action_max - env_params.action_min + 1
        state_dim = env_params.observation_max - env_params.observation_min + 1

        # Configure the qtable
        self.qtable = QTable(state_dim=state_dim, action_dim=action_dim, batch_size=1, initial_value=0.0, track_visits=True, device='cpu')

        if decision_function is None:
            self.decision_function = EpsilonGreedy(epsilon=0.1)
        else:
            self.decision_function = decision_function

    def to(self, device: str) -> None:
        """
        Overrides the default to method to move the Q-table to the specified device.
        
        Args:
            device (str): The device to move the Q-table to ('cpu' or 'cuda').
        """
        self.qtable.to(device)

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
                - value_estimate (torch.Tensor): None
                - log_prob (torch.Tensor): None
        """
        # Get the action values for the current state
        action_vals = self.qtable.get_action_values(state)

        if not deterministic:
            action = self.decision_function.select_action(action_vals)
        else:
            action = torch.argmax(action_vals, dim=-1, keepdim=True)
        return action, None, None
    
    def get_action_values(self,
                          state: torch.Tensor
                          ) -> torch.Tensor:
        """
        Returns the action values for the given state.
        Args:
            state (torch.Tensor): Current state tensor.
        Returns:
            torch.Tensor: Tensor with action values.
        """
        return self.qtable.get_action_values(state)
    
    def get_state_action_value(self,
                                 state: torch.Tensor,
                                 action: torch.Tensor
                                 ) -> torch.Tensor:
        """
        Returns the Q-value for the given state-action pair.

        Args:
                state (torch.Tensor): State tensor.
                action (torch.Tensor): Action tensor.
        Returns:
                torch.Tensor: Q-value for the given state-action pair.
        """
        return self.qtable.get_state_action_value(state, action)
    
    def get_visit_count(self,
                        state: torch.Tensor,
                        action: torch.Tensor
                        ) -> torch.Tensor:
        """
        Returns the visit count for the given state-action pair.
        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: Visit count for the given state-action pair.
        """        
        return self.qtable.get_visit_count(state, action)
    
    def update_q_value(self,
                       state: torch.Tensor,
                       action: torch.Tensor,
                       q_value: torch.Tensor
                       ) -> None:
        """
        Updates the Q table for a given state-action pair with given q-value.
        Args:
            state (torch.Tensor): state value to update the Q table for with shape (# env, 1)
            action (torch.Tensor): action value to update the Q table for with shape (# env, 1)
            q_value (torch.Tensor): q-value to update the Q table for with shape (# env, 1)

        """
        self.qtable.update_q_value(state, action, q_value)    

    def update_visits(self,
                      state: torch.Tensor,
                      action: torch.Tensor
                      ) -> None:
        """
        Updates the visit count for the given state-action pair.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        """
        self.qtable.update_visits(state, action)