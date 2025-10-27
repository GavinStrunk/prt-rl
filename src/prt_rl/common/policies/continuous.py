import torch
from typing import Optional, Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies.base import BasePolicy
from prt_rl.common.networks import MLP, BaseEncoder
from prt_rl.common.utils import clamp_actions


class ContinuousPolicy(BasePolicy):
    """
    ContinuousPolicy is a policy that uses a neural network to compute actions for continuous action spaces. It can optionally use an encoder network to process the input state before passing it to the policy head.

    .. note::
        The ContinuousPolicy always returns a deterministic action based on the current state.
    
    The architecture of the policy is as follows:
        - Encoder Network (optional): Processes the input state.
        - Policy Head: Computes actions based on the latent state.

    .. image:: /_static/continuouspolicy.png
        :alt: ContinuousPolicy Architecture
        :width: 100%
        :align: center

    Args:
        env_params (EnvParams): Environment parameters.
        encoder_network (BaseEncoder | None): Encoder network to process the input state. If None, the input state is used directly.
        encoder_network_kwargs (Optional[dict]): Keyword arguments for the encoder network.
        policy_head (torch.nn.Module): Policy head network to compute actions. Default is MLP.
        policy_head_kwargs (Optional[dict]): Keyword arguments for the policy head network.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder_network: BaseEncoder | None = None,
                 encoder_network_kwargs: Optional[dict] = {},                 
                 policy_head: torch.nn.Module = MLP,
                 policy_head_kwargs: Optional[dict] = {},
                 ) -> None:
        super().__init__(env_params)
        if not env_params.action_continuous:
            raise ValueError("ContinuousPolicy only supports continuous action spaces. Use a different policy class.")
        
        if encoder_network is None:
            self.encoder = encoder_network
            latent_dim = env_params.observation_shape[0]
        else:
            self.encoder = encoder_network(
                input_shape=env_params.observation_shape,
                **encoder_network_kwargs
                )
            latent_dim = self.encoder.features_dim

        self.policy_head = policy_head(
            input_dim=latent_dim,
            output_dim=env_params.action_len,
           **policy_head_kwargs
        )        
    
    def predict(self, 
                state: torch.Tensor, 
                deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chooses an action based on the current state and returns the action, value estimate, and log probability.

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): This value is ignored as the policy always returns a deterministic action.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the chosen action, value estimate, and action log probability.
                - action (torch.Tensor): Tensor with the chosen action. Shape (B, action_dim)
                - value_estimate (torch.Tensor): None
                - log_prob (torch.Tensor): None
        """
        if self.encoder is not None:
            state = self.encoder(state)

        action = self.policy_head(state)

        action = clamp_actions(action, self.env_params.action_min, self.env_params.action_max)
        return action, None, None

    
    def get_encoder(self) -> Optional[BaseEncoder]:
        """
        Returns the encoder network used by the policy. 
        Returns:
            Optional[BaseEncoder]: The encoder network if it exists, otherwise None.
        """
        return self.encoder