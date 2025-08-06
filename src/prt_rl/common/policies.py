"""
Summary of Policy Architectures

.. image:: /_static/qvaluepolicy.png
    :alt: QValuePolicy Architecture
    :width: 100%
    :align: center

.. image:: /_static/distributionpolicy.png
    :alt: DistributionPolicy Architecture
    :width: 100%
    :align: center

.. image:: /_static/actorcriticpolicy.png
    :alt: ActorCriticPolicy Architecture
    :width: 100%
    :align: center

Critic Architectures
.. image:: /_static/valuecritic.png
    :alt: ValueCritic Architecture
    :width: 100%
    :align: center

.. image:: /_static/stateactioncritic.png
    :alt: StateActionCritic Architecture
    :width: 100%
    :align: center
"""
from abc import ABC, abstractmethod
import copy
import torch
from typing import Optional, Union, Dict, Type, Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.decision_functions import DecisionFunction, EpsilonGreedy
from prt_rl.common.networks import MLP, BaseEncoder
import prt_rl.common.distributions as dist


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

    @abstractmethod
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
        raise NotImplementedError

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
        encoder_network (Optional[Type[BaseEncoder]]): Encoder network to process the input state. If None, the input state is used directly.
        encoder_network_kwargs (Optional[dict]): Keyword arguments for the encoder network.
        policy_head (Optional[Type[torch.nn.Module]]): Policy head network to compute Q-values. Default is MLP.
        policy_head_kwargs (Optional[dict]): Keyword arguments for the policy head network.
        decision_function (Optional[DecisionFunction]): Decision function to select actions based on Q-values. Default is EpsilonGreedy.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder_network: Optional[Type[BaseEncoder]] = None,
                 encoder_network_kwargs: Optional[dict] = {},
                 policy_head: Optional[Type[torch.nn.Module]] = MLP,
                 policy_head_kwargs: Optional[dict] = {},
                 decision_function: Optional[DecisionFunction] = None,
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
        if env_params.action_continuous:
            action_dim = env_params.action_len
        else:
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

    def forward(self,
                   state: torch.Tensor,
                   deterministic: bool = False
                   ) -> torch.Tensor:
        """
        Chooses an action based on the current state. Expects the key "observation" in the state tensordict.

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.

        Returns:
            torch.Tensor: Tensor with the chosen action.
        """
        q_vals = self.get_q_values(state)

        with torch.no_grad():
            if not deterministic:
                action = self.decision_function.select_action(q_vals)
            else:
                action = torch.argmax(q_vals, dim=-1, keepdim=True)
        return action
    
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
        encoder_network (Optional[Union[Type[BaseEncoder], Dict[str, Type[BaseEncoder]]]]): Encoder network to process the input state. If None, the input state is used directly.
        encoder_network_kwargs (Optional[dict]): Keyword arguments for the encoder network.
        policy_head (Union[Type[torch.nn.Module], Dict[str, Type[torch.nn.Module]]]): Policy head network to compute latent features. Default is MLP.
        policy_kwargs (Optional[dict]): Keyword arguments for the policy head network.
        distribution (Optional[dist.Distribution]): Distribution to use for the policy. If None, defaults to Categorical for discrete action spaces and Normal for continuous action spaces.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder_network: Optional[Union[Type[BaseEncoder], Dict[str, Type[BaseEncoder]]]] = None,
                 encoder_network_kwargs: Optional[dict] = {},
                 policy_head: Union[Type[torch.nn.Module], Dict[str, Type[torch.nn.Module]]] = MLP,
                 policy_kwargs: Optional[dict] = {},
                 distribution: Optional[dist.Distribution] = None,
                 ) -> None:
        super().__init__(env_params=env_params)
        self.env_params = env_params
        self.encoder_network = None

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

    def forward(self,
                   state: torch.Tensor,
                   deterministic: bool = False
                   ) -> torch.Tensor:
        """
        Chooses an action based on the current state

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.
        Returns:
            torch.Tensor: Tensor with the chosen action with shape (N, action_dim)
        """
        action, _ = self.predict(state, deterministic=deterministic)
        return action
    
    def predict(self,
                   state: torch.Tensor,
                   deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chooses an action based on the current state. 

        state -> Encoder Network -> Policy Head -> Distribution Layer -> Distribution ..
        .. -> Sample -> Action
        .. -> Log Probabilities
        

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.

        Returns:
            torch.Tensor: Tensor with the chosen action. (N, action_dim), (N, # actions)
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

        log_probs = distribution.log_prob(action)

        # Compute the total log probability for the action vector
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return action, log_probs
    
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
        encoder_network (Optional[Union[Type[BaseEncoder], Dict[str, Type[BaseEncoder]]]]): Encoder network to process the input state. If None, the input state is used directly.
        encoder_network_kwargs (Optional[dict]): Keyword arguments for the encoder network.
        policy_head (Type[torch.nn.Module]): Policy head network to compute actions. Default is MLP.
        policy_head_kwargs (Optional[dict]): Keyword arguments for the policy head network.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder_network: Optional[Union[Type[BaseEncoder], Dict[str, Type[BaseEncoder]]]] = None,
                 encoder_network_kwargs: Optional[dict] = {},                 
                 policy_head: Type[torch.nn.Module] = MLP,
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


    def forward(self,
                   state: torch.Tensor,
                   deterministic: bool = False
                   ) -> torch.Tensor:
        """
        Chooses an action based on the current state.

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): This value is ignored as the policy always returns a deterministic action.

        Returns:
            torch.Tensor: Tensor with the chosen action.
        """
        if self.encoder is not None:
            state = self.encoder(state)

        action = self.policy_head(state)
        return action.clamp(self.env_params.action_min, self.env_params.action_max)

class ValueCritic(torch.nn.Module):
    """
    ValueCritic is a critic network that estimates the value of a given state.

    The architecture of the critic is as follows:
        - Encoder Network (optional): Processes the input state.
        - Critic Head: Computes the value for the given state.

    .. image:: /_static/valuecritic.png
        :alt: ValueCritic Architecture
        :width: 100%
        :align: center
    
    Args:
        env_params (EnvParams): Environment parameters.
        encoder (torch.nn.Module | None): Encoder network to process the input state. If None, the input state is used directly.
        critic_head (Type[torch.nn.Module]): Critic head network to compute values. Default is MLP.
        critic_head_kwargs (Optional[dict]): Keyword arguments for the critic head network.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder: torch.nn.Module | None = None,
                 critic_head: Type[torch.nn.Module] = MLP,
                 critic_head_kwargs: Optional[dict] = {},                 
                 ) -> None:
        super().__init__()
        self.env_params = env_params
        self.encoder = encoder

        if self.encoder is not None:
            latent_dim = self.encoder.features_dim
        else:
            latent_dim = self.env_params.observation_shape[0]

        self.critic_head = critic_head(
            input_dim=latent_dim,
            output_dim=1,
            **critic_head_kwargs
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.

        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            torch.Tensor: The estimated value for the given state.
        """
        if self.encoder is not None:
            state = self.encoder(state)

        return self.critic_head(state)


class StateActionCritic(torch.nn.Module):
    """
    StateActionCritic is a critic network that takes both state and action as input and outputs the Q-value for the given state-action pair. It can handle multiple critics for ensemble methods.

    .. note::
        If multiple critics are used and an encoder is provided, the encoder will be shared across all critics. If no encoder is provided, the input state is used directly.

    The architecture of the critic is as follows:
        - Encoder Network (optional): Processes the input state.
        - Critic Head: Computes Q-values for the given state-action pair.

    .. image:: /_static/stateactioncritic.png
        :alt: StateActionCritic Architecture
        :width: 100%
        :align: center
    
    Args:
        env_params (EnvParams): Environment parameters.
        num_critics (int): Number of critics to use. Default is 1.
        encoder (torch.nn.Module | None): Encoder network to process the input state. If None, the input state is used directly.
        critic_head (Type[torch.nn.Module]): Critic head network to compute Q-values. Default is MLP.
        critic_head_kwargs (Optional[dict]): Keyword arguments for the critic head network.
    """
    def __init__(self, 
                 env_params: EnvParams, 
                 num_critics: int = 1,
                 encoder: torch.nn.Module | None = None,
                 critic_head: Type[torch.nn.Module] = MLP,
                 critic_head_kwargs: Optional[dict] = {},
                 ) -> None:
        super(StateActionCritic, self).__init__()
        self.env_params = env_params
        self.num_critics = num_critics
        self.encoder = encoder

        if self.encoder is not None:
            latent_dim = self.encoder.features_dim
        else:
            latent_dim = self.env_params.observation_shape[0]

        # Initialize critics here
        self.critics = []
        for _ in range(num_critics):
            critic = critic_head(
                input_dim=latent_dim + self.env_params.action_len,
                output_dim=1,
                **critic_head_kwargs
            )
            self.critics.append(critic)

        # Convert list to ModuleList for proper parameter management
        self.critics = torch.nn.ModuleList(self.critics)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor] | torch.Tensor:
        """
        Forward pass through the critic network.

        Args:
            state: The current state of the environment.
            action: The action taken in the current state.

        Returns:
            A tuple of Q-values for the given state-action pair for all critics.
        """
        if self.encoder is not None:
            state = self.encoder(state)

        # Stack the state and action tensors
        q_input = torch.cat([state, action], dim=1)

        # Return a tuple of Q-values from each critic or a single tensor if only one critic is used
        if self.num_critics == 1:
            return self.critics[0](q_input)
        else:
            return tuple(critic(q_input) for critic in self.critics)

    def forward_indexed(self, index: int, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network at the index provided.

        Args:
            index (int): The index of the critic to use.
            state (torch.Tensor): The current state of the environment.
            action (torch.Tensor): The action taken in the current state.

        Returns:
            The Q-value for the given state-action pair from the first critic.
        """
        if index > self.num_critics - 1:
            raise ValueError(f"Index {index} exceeds the number of critics {self.num_critics}.")
        
        if self.encoder is not None:
            state = self.encoder(state)

        # Stack the state and action tensors
        q_input = torch.cat([state, action], dim=1)
        return self.critics[index](q_input)

class ActorCriticPolicy(BasePolicy):
    """
    
    This policy assumes if you provide a single encoder network, it will be shared between the actor and critic. It also assumes if a single network is provided for the actor and critic heads, it will be shared between them. If you want to use different networks for the actor and critic, you can provide them separately.
    The policy head network should only define up to the last feature layer of the network. The specific distribution initializes the final layer of the network to ensure it is compatible.

    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder_network: Optional[Union[Type[BaseEncoder], Dict[str, Type[BaseEncoder]]]] = None,
                 encoder_network_kwargs: Optional[dict] = {},
                 actor_critic_head: Union[Type[torch.nn.Module], Dict[str, Type[torch.nn.Module]]] = MLP,
                 actor_critic_head_kwargs: Optional[dict] = {},
                 distribution: Optional[dist.Distribution] = None,
                 device: str = "cpu",
                 ) -> None:
        super().__init__(env_params=env_params)
        self.device = device
        self.env_params = env_params

        self._build_encoder(encoder_network, encoder_network_kwargs)
        self.actor_feature_dim = self._build_actor_critic_head(actor_critic_head, actor_critic_head_kwargs)
        self._build_distribution(distribution)

            
    def _build_encoder(self, 
                       encoder_network: Union[Type[BaseEncoder], Dict[str, Type[BaseEncoder]], None], 
                       encoder_network_kwargs: dict
                       ) -> None:
        """
        Builds the encoder network for the policy.
        Args:
            encoder_network (torch.nn.Module or Dict[str, torch.nn.Module]): The encoder network or a dictionary of encoder networks for actor and critic.
            encoder_network_kwargs (dict): Keyword arguments for the encoder network.
        """
        # Initialize Type 1: No Encoder Network
        if encoder_network is None:
            self.actor_encoder_network = None
            self.critic_encoder_network = None
            self.actor_latent_dim = self.env_params.observation_shape[0]
            self.critic_latent_dim = self.env_params.observation_shape[0]

        # Initialize Type 3: Construct encoder networks when they are separate
        elif isinstance(encoder_network, dict):
            if 'actor' not in encoder_network or 'critic' not in encoder_network:
                raise ValueError("If encoder_network is a dictionary, it must contain keys 'actor' and 'critic'.")
            
            if 'actor' not in encoder_network_kwargs or 'critic' not in encoder_network_kwargs:
                raise ValueError("If encoder_network is a dictionary, encoder_network_kwargs must contain keys 'actor' and 'critic'.")
            
            self.actor_encoder_network = encoder_network['actor'](
                    input_shape=self.env_params.observation_shape,
                    **encoder_network_kwargs['actor']
                )
            self.critic_encoder_network = encoder_network['critic'](
                    input_shape=self.env_params.observation_shape,
                    **encoder_network_kwargs['critic']
                )
            self.actor_latent_dim = self.actor_encoder_network.features_dim
            self.critic_latent_dim = self.critic_encoder_network.features_dim

        # Initialize Type 3: Construct encoder networks when they are shared
        elif issubclass(encoder_network, BaseEncoder):
            self.actor_encoder_network = encoder_network(
                    input_shape=self.env_params.observation_shape,
                    **encoder_network_kwargs
                )
            self.critic_encoder_network = self.actor_encoder_network
            self.actor_latent_dim = self.actor_encoder_network.features_dim
            self.critic_latent_dim = self.actor_encoder_network.features_dim
        else:
            raise ValueError("encoder_network must be either None, a BaseEncoder, or a dictionary with keys 'actor' and 'critic'.")
    
    def _build_actor_critic_head(self,
                                 actor_critic_head: Union[Type[torch.nn.Module], Dict[str, Type[torch.nn.Module]]],
                                 actor_critic_head_kwargs: dict,
                                 ) -> None:
        """
        Builds the actor and critic heads for the policy.
        Args:
            actor_critic_head (torch.nn.Module or Dict[str, torch.nn.Module]): The actor and critic heads or a dictionary of actor and critic heads.
            actor_critic_head_kwargs (dict): Keyword arguments for the actor and critic heads.
        """
        # Initialize Type 1: Construct separate actor and critic heads
        if isinstance(actor_critic_head, dict):
            if 'actor' not in actor_critic_head or 'critic' not in actor_critic_head:
                raise ValueError("If actor_critic_head is a dictionary, it must contain keys 'actor' and 'critic'.")
            if 'actor' not in actor_critic_head_kwargs or 'critic' not in actor_critic_head_kwargs:
                raise ValueError("If actor_critic_head is a dictionary, actor_critic_head_kwargs must contain keys 'actor' and 'critic'.")
            
            self.actor_head = actor_critic_head['actor'](
                input_dim=self.actor_latent_dim,
                **actor_critic_head_kwargs['actor']
            )
            self.critic_head = actor_critic_head['critic'](
                input_dim=self.critic_latent_dim,
                output_dim=1,
                **actor_critic_head_kwargs['critic']
            )
        
        # Initialize Type 2: Construct the same network for actor and critic heads
        elif issubclass(actor_critic_head, torch.nn.Module):
            self.actor_head = actor_critic_head(
                input_dim=self.actor_latent_dim,
                **actor_critic_head_kwargs
            )

            # Set the 'output_dim' key to 1 for the critic head
            self.critic_head = actor_critic_head(
                input_dim=self.critic_latent_dim,
                output_dim=1,
                **actor_critic_head_kwargs
            )   
        else:
            raise ValueError("actor_critic_head must be either a torch.nn.Module, or a dictionary with keys 'actor' and 'critic'.")                              
        
        # Last layer is an activation so we can get the feature dimension from the second to last linear layer
        return self.actor_head.layers[-2].out_features

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
        self.actor_distribution_layer = self.distribution.last_network_layer(feature_dim=self.actor_feature_dim, action_dim=action_dim)


    def forward(self,
                   state: torch.Tensor
                   ) -> torch.Tensor:
        """
        Chooses an action based on the current state. Expects the key "observation" in the state tensordict

        Args:
            state (TensorDict): current state tensordict

        Returns:
            TensorDict: tensordict with the "action" key added
        """
        action, _, _ = self.predict(state)
        return action

    def predict(self,
                state: torch.Tensor
                ) -> torch.Tensor:
        # Run Actor
        if self.actor_encoder_network is None:
            action_encoding = state
        else:
            action_encoding = self.actor_encoder_network(state)

        latent_features = self.actor_head(action_encoding)
        action_params = self.actor_distribution_layer(latent_features)
        distribution = self.distribution(action_params)
        action = distribution.sample()
        log_probs = distribution.log_prob(action)

        # Convert the action and log probabilities to the correct shape (N, ) -> (N, 1)
        action = action.unsqueeze(-1)
        log_probs = log_probs.unsqueeze(-1)

        # Run Critic
        if self.critic_encoder_network is None:
            critic_features = state
        else:
            critic_features = self.critic_encoder_network(state)
            
        value_est = self.critic_head(critic_features)

        return action, value_est, log_probs
    
    def evaluate_actions(self,
                         state: torch.Tensor,
                         action: torch.Tensor
                         ) -> torch.Tensor:
        # Run Actor
        if self.actor_encoder_network is None:
            action_encoding = state
        else:
            action_encoding = self.actor_encoder_network(state)

        latent_features = self.actor_head(action_encoding)
        action_params = self.actor_distribution_layer(latent_features)
        distribution = self.distribution(action_params)
        entropy = distribution.entropy().unsqueeze(-1)
        log_probs = distribution.log_prob(action.squeeze()).unsqueeze(-1)

        # Run Critic
        if self.critic_encoder_network is None:
            critic_features = state
        else:
            critic_features = self.critic_encoder_network(state)

        value_est = self.critic_head(critic_features)

        return value_est, log_probs, entropy

