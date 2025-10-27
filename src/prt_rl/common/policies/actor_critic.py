import copy
import torch
from typing import Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.networks import BaseEncoder
from prt_rl.common.policies.base import BasePolicy
from prt_rl.common.policies.distribution import DistributionPolicy
from prt_rl.common.policies.value_critic import ValueCritic


class ActorCriticPolicy(BasePolicy):
    """
    ActorCriticPolicy is a policy that combines an actor and a critic network. It can optionally use an encoder network to process the input state before passing it to the actor and critic heads.

    The ActorCriticPolicy is a combination of a DistributionPolicy for the actor and a ValueCritic for the critic. It can handle both discrete and continuous action spaces.
    
    The architecture of the policy is as follows:
        - Encoder Network (optional): Processes the input state.
        - Actor Head: Computes actions based on the latent state.
        - Critic Head: Computes the value for the given state.

    .. image:: /_static/actorcriticpolicy.png
        :alt: ActorCriticPolicy Architecture
        :width: 100%
        :align: center

    Args:
        env_params (EnvParams): Environment parameters.
        encoder (BaseEncoder | None): Encoder network to process the input state. If None, the input state is used directly.
        actor (DistributionPolicy | None): Actor network to compute actions. If None, a default DistributionPolicy is created.
        critic (ValueCritic | None): Critic network to compute values. If None, a default ValueCritic is created.
        share_encoder (bool): If True, share the encoder between actor and critic. Default is False.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder: BaseEncoder | None = None,
                 actor: DistributionPolicy | None = None,
                 critic: ValueCritic | None = None,
                 share_encoder: bool = False,
                 ) -> None:
        super().__init__(env_params=env_params)
        self.env_params = env_params
        self.encoder = encoder
        self.critic_encoder = None
        self.share_encoder = share_encoder
        
        # If no actor is provided, create a default DistributionPolicy without an encoder
        if actor is None:
            self.actor = DistributionPolicy(
                env_params=env_params,
            )
        else:
            self.actor = actor

        # If no critic is provided, create a default ValueCritic without an encoder
        if critic is None:
            self.critic = ValueCritic(
                env_params=env_params,
            )
        else:
            self.critic = critic

        # If the encoder is not shared, but one exists then make a copy for the critic
        if not share_encoder and self.encoder is not None:
            self.critic_encoder = copy.deepcopy(self.encoder)  
    
    def predict(self,
                   state: torch.Tensor,
                   deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chooses an action based on the current state and computes the value of the state.

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the chosen action, value_estimate, and aciton log probability.
        """
        if self.encoder is not None:
            latent_state = self.encoder(state)
        else:
            latent_state = state

        action, _, log_probs = self.actor.predict(latent_state, deterministic=deterministic)

        if self.critic_encoder is not None:
            critic_latent_state = self.critic_encoder(state)
        else:
            critic_latent_state = latent_state

        value = self.critic(critic_latent_state) 
        return action, value, log_probs
    
    def evaluate_actions(self,
                         state: torch.Tensor,
                         action: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the value, log probability and entropy of the given action under the policy.
        Args:
            state (torch.Tensor): Current state tensor.
            action (torch.Tensor): Action tensor to evaluate.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the value estimate, log probability, and entropy. All tensors have shape (B, 1).
        """
        if self.encoder is not None:
            latent_state = self.encoder(state)
        else:
            latent_state = state
        
        log_probs, entropy = self.actor.evaluate_actions(latent_state, action)

        if self.critic_encoder is not None:
            critic_latent_state = self.critic_encoder(state)
        else:
            critic_latent_state = latent_state
        
        value = self.critic(critic_latent_state)

        return value, log_probs, entropy