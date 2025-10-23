import torch
from typing import Optional, Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies.base import BasePolicy
from prt_rl.common.networks import MLP, BaseEncoder
import prt_rl.common.distributions as dist


class DistributionPolicy(BasePolicy):
    """
    The DistributionPolicy class implements a policy that uses a neural network to compute action distributions for both discrete and continuous action spaces. It can optionally use an encoder network to process the input state before passing it to the policy head.

    The architecture of the policy is as follows:
        - Encoder Network (optional): Processes the input state.
        - Policy Head: Computes latent features from the encoded state.
        - Distribution Layer: Maps the latent features to action distributions.

    .. image:: /_static/distributionpolicy.png
        :alt: DistributionPolicy Architecture
        :width: 100%
        :align: center

    Args:
        env_params (EnvParams): Environment parameters.
        encoder_network (BaseEncoder | None): Encoder network to process the input state. If None, the input state is used directly.
        encoder_network_kwargs (Optional[dict]): Keyword arguments for the encoder network.
        policy_head (torch.nn.Module): Policy head network to compute latent features. Default is MLP.
        policy_kwargs (Optional[dict]): Keyword arguments for the policy head network.
        distribution (dist.Distribution | None): Distribution to use for the policy. If None, defaults to Categorical for discrete action spaces and Normal for continuous action spaces.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder_network: BaseEncoder | None = None,
                 encoder_network_kwargs: Optional[dict] = {},
                 policy_head: torch.nn.Module = MLP,
                 policy_kwargs: Optional[dict] = {},
                 distribution: dist.Distribution | None = None,
                 return_log_prob: bool = True
                 ) -> None:
        super().__init__(env_params=env_params)
        self.env_params = env_params
        self.encoder_network = None
        self.return_log_prob = return_log_prob

        # Construct the encoder network if provided
        if encoder_network is not None:
            self.encoder_network = encoder_network(
                input_shape=self.env_params.observation_shape,
                **encoder_network_kwargs
            )
            self.latent_dim = self.encoder_network.features_dim
        else:
            self.encoder_network = None
            self.latent_dim = self.env_params.observation_shape[0]
        
        # Construct the policy head network
        self.policy_head = policy_head(
            input_dim=self.latent_dim,
            **policy_kwargs
        )

        self.policy_feature_dim = self.policy_head.layers[-2].out_features

        self._build_distribution(distribution)

        # Build the distribution layer
    def _build_distribution(self,
                           distribution: dist.Distribution,
                           ) -> None:
        """
        Builds the distribution for the policy.

        Args:
            distribution (dist.Distribution): The distribution to use for the policy.
        """
        # Default distributions for discrete and continuous action spaces
        if distribution is None:
            if self.env_params.action_continuous:
                self.distribution = dist.Normal
            else:
                self.distribution = dist.Categorical
        else:
            self.distribution = distribution

        action_dim = self.distribution.get_action_dim(self.env_params)

        dist_layer = self.distribution.last_network_layer(feature_dim=self.policy_feature_dim, action_dim=action_dim)

        # Support both interfaces: torch.nn.Module and Tuple[torch.nn.Module, torch.nn.Parameter]
        if isinstance(dist_layer, tuple):
            self.distribution_layer = dist_layer[0]
            self.distribution_params = dist_layer[1]
        else:
            self.distribution_layer = dist_layer
            self.distribution_params = None
    
    def predict(self,
                   state: torch.Tensor,
                   deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chooses an action based on the current state. 

        state -> Encoder Network -> Policy Head -> Distribution Layer -> Distribution ..
        .. -> Sample -> Action
        .. -> Log Probabilities
        

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the chosen action, value estimate, and action log probability.
                - action (torch.Tensor): Tensor with the chosen action. Shape (B, action_dim)
                - value_estimate (torch.Tensor): None
                - log_prob (torch.Tensor): Tensor with the log probability of the chosen action. Shape (B, 1)
        """
        if self.encoder_network is not None:
            latent_state = self.encoder_network(state)
        else:
            latent_state = state
        
        latent_features = self.policy_head(latent_state)
        dist_params = self.distribution_layer(latent_features)

        # If the distribution has parameters, we use them to create the distribution
        if self.distribution_params is not None:
            distribution = self.distribution(dist_params, self.distribution_params)
        else:
            distribution = self.distribution(dist_params)

        if deterministic:
            action = distribution.deterministic_action()
        else:
            action = distribution.sample()

        if self.return_log_prob:
            log_probs = distribution.log_prob(action)

            # Compute the total log probability for the action vector
            log_probs = log_probs.sum(dim=-1, keepdim=True)
        else:
            log_probs = None

        return action, None, log_probs
    
    def evaluate_actions(self,
                         state: torch.Tensor,
                         action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the log probability and entropy of the given action under the policy.

        Args:
            state (torch.Tensor): Current state tensor.
            action (torch.Tensor): Action tensor to evaluate.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the log probability of the action and the entropy of the distribution. 
                - log_prob (torch.Tensor): Tensor with the log probability of the given action. Shape (B, 1)
                - entropy (torch.Tensor): Tensor with the entropy of the distribution. Shape (B, 1)
        """
        if self.encoder_network is not None:
            latent_state = self.encoder_network(state)
        else:
            latent_state = state
        
        latent_features = self.policy_head(latent_state)
        dist_params = self.distribution_layer(latent_features)

        # If the distribution has parameters, we use them to create the distribution
        if self.distribution_params is not None:
            distribution = self.distribution(dist_params, self.distribution_params)
        else:
            distribution = self.distribution(dist_params)

        # Compute log probabilities and entropy for the entire action vector
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        log_probs = distribution.log_prob(action.squeeze()).sum(dim=-1, keepdim=True)

        return log_probs, entropy
    
    def get_logits(self,
                        state: torch.Tensor
                    ) -> torch.Tensor:
        """
        Returns the logits from the policy network given the input state.

        state -> Encoder Network -> Policy Head -> Categorical Layer -> logits

        Args:
            state (torch.Tensor): Input state tensor of shape (N, obs_dim).

        Returns:
            torch.Tensor: Logits tensor of shape (N, num_actions).
        """
        if not issubclass(self.distribution, dist.Categorical):
            raise ValueError("get_logits is only supported for Categorical distributions. Use forward for other distributions.")
        
        if self.encoder_network is not None:
            latent_state = self.encoder_network(state)
        else:
            latent_state = state
        
        latent_features = self.policy_head(latent_state)
        logits = self.distribution_layer(latent_features)
        return logits
    
    def get_encoder(self) -> Optional[BaseEncoder]:
        """
        Returns the encoder network used by the policy.

        Returns:
            Optional[BaseEncoder]: The encoder network if it exists, otherwise None.
        """
        return self.encoder_network