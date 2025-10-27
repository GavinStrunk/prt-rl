import torch
from typing import Optional, Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies.base import BasePolicy
from prt_rl.common.decision_functions import DecisionFunction, EpsilonGreedy
from prt_rl.common.networks import MLP, BaseEncoder


class QValuePolicy(BasePolicy):
    """
    The QValuePolicy class implements a policy that uses a neural network to compute Q-values for discrete action spaces. It can optionally use an encoder network to process the input state before passing it to the policy head.

    The architecture of the policy is as follows:
        - Encoder Network (optional): Processes the input state.
        - Policy Head: Computes Q-values for each action based on the latent state.
        - Decision Function: Selects actions based on the Q-values.

    .. note::
        This policy is designed for discrete action spaces. For continuous action spaces, use a different policy class.

    .. image:: /_static/qvaluepolicy.png
        :alt: QValuePolicy Architecture
        :width: 100%
        :align: center

    Args:
        env_params (EnvParams): Environment parameters.
        encoder_network (BaseEncoder | None): Encoder network to process the input state. If None, the input state is used directly.
        encoder_network_kwargs (Optional[dict]): Keyword arguments for the encoder network.
        policy_head (torch.nn.Module): Policy head network to compute Q-values. Default is MLP.
        policy_head_kwargs (Optional[dict]): Keyword arguments for the policy head network.
        decision_function (DecisionFunction | None): Decision function to select actions based on Q-values. Default is EpsilonGreedy.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder_network: BaseEncoder | None = None,
                 encoder_network_kwargs: Optional[dict] = {},
                 policy_head: torch.nn.Module = MLP,
                 policy_head_kwargs: Optional[dict] = {},
                 decision_function: DecisionFunction | None = None,
                 ) -> None:
        super().__init__(env_params)

        if env_params.action_continuous:
            raise ValueError("QValuePolicy does not support continuous action spaces. Use a different policy class.")
        
        if encoder_network is None:
            self.encoder_network = encoder_network
            latent_dim = env_params.observation_shape[0]
        else:
            self.encoder_network = encoder_network(
                input_shape=env_params.observation_shape,
                **encoder_network_kwargs
                )
            latent_dim = self.encoder_network.features_dim

        # Get action dimension
        action_dim = env_params.action_max - env_params.action_min + 1

        self.policy_head = policy_head(
            input_dim=latent_dim,
            output_dim=action_dim,
           **policy_head_kwargs
        )

        if decision_function is None:
            self.decision_function = EpsilonGreedy(epsilon=1.0)
        else:
            self.decision_function = decision_function
    
    def predict(self, 
                state: torch.Tensor, 
                deterministic = False
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
        value_est = self.get_q_values(state)

        if not deterministic:
            action = self.decision_function.select_action(value_est)
        else:
            action = torch.argmax(value_est, dim=-1, keepdim=True)

        return action, None, None
    
    def get_q_values(self,
                        state: torch.Tensor
                    ) -> torch.Tensor:
        """
        Returns the action probabilities for the given state.

        Args:
            state (torch.Tensor): Current state tensor.

        Returns:
            torch.Tensor: Tensor with action probabilities.
        """
        if self.encoder_network is not None:
            latent_state = self.encoder_network(state)
        else:
            latent_state = state

        q_vals = self.policy_head(latent_state)
        return q_vals
    
    def get_encoder(self) -> Optional[BaseEncoder]:
        """
        Returns the encoder network used by the policy.

        Returns:
            Optional[BaseEncoder]: The encoder network if it exists, otherwise None.
        """
        return self.encoder_network 