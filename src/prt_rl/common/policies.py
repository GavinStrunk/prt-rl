from abc import ABC, abstractmethod
import torch
from typing import Any, Optional
# from tensordict.tensordict import TensorDict
from prt_rl.env.interface import EnvParams
from prt_rl.common.decision_functions import DecisionFunction, EpsilonGreedy
from prt_rl.common.networks import MLP
import prt_rl.common.distributions as dist

# def load_from_mlflow(
#         tracking_uri: str,
#         model_name: str,
#         model_version: str,
# ) -> 'Policy':
#     """
#     Loads a model that is either registered in mlflow or associated with a run id.

#     Args:
#         tracking_uri (str): mlflow tracking uri
#         model_name (str): name of the model in the registry
#         model_version (str): string version of the model

#     Returns:
#         Policy: policy object
#     """
#     try:
#         import mlflow
#     except ModuleNotFoundError:
#         raise ModuleNotFoundError("mlflow is required to be installed load a policy from mlflow")

#     mlflow.set_tracking_uri(tracking_uri)
#     client = mlflow.tracking.MlflowClient()
#     registered_models = client.search_registered_models()
#     for model in registered_models:
#         print(f"Model Name: {model.name}")

#     model_str = f"models:/{model_name}/{model_version}"
#     policy = mlflow.pyfunc.load_model(model_uri=model_str)

#     # Extract the metadata
#     metadata = policy.metadata.metadata

#     # Policy factory
#     module_name = f"prt_rl.utils.policy"
#     try:
#         module = importlib.import_module(module_name)
#         policy_class = getattr(module, metadata['type'])
#         policy = policy_class.load_from_dict(metadata['policy'])
#     except ModuleNotFoundError:
#         raise ValueError(f"Class {metadata['type']} not found")

#     # if metadata['type'] == 'QTablePolicy':
#     #     return QTablePolicy.load_from_dict(metadata['policy'])
#     return policy

class BasePolicy(torch.nn.Module, ABC):
    """
    Base class for implementing policies.

    Args:
        env_params (EnvParams): Environment parameters.
        device (str): The device to use.
    """
    def __init__(self,
                 env_params: EnvParams,
                 ) -> None:
        super().__init__()
        self.env_params = env_params

    def __call__(self,
                   state: torch.Tensor
                   ) -> torch.Tensor:
        return self.forward(state)

    @abstractmethod
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
        raise NotImplementedError

class QValuePolicy(BasePolicy):
    """
    Base class for implementing discrete policies.

    Args:
        env_params (EnvParams): Environment parameters.
        device (str): The device to use.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder_network: Optional[torch.nn.Module] = None,
                 encoder_network_kwargs: Optional[dict] = {},
                 policy_head: Optional[torch.nn.Module] = MLP,
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
                   state: torch.Tensor
                   ) -> torch.Tensor:
        """
        Chooses an action based on the current state. Expects the key "observation" in the state tensordict.

        Args:
            state (torch.Tensor): Current state tensor.

        Returns:
            torch.Tensor: Tensor with the chosen action.
        """
        q_vals = self.get_q_values(state)
        with torch.no_grad():
            action = self.decision_function.select_action(q_vals)
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


# class ActorCriticPolicy(Policy):
#     """
#     Actor critic policy
#     """
#     def __init__(self,
#                  env_params: EnvParams,
#                  distribution: Optional[dist.Distribution] = None,
#                  device: str = "cpu",
#                  ) -> None:
#         super().__init__(env_params=env_params, device=device)
#         self.distribution = distribution

#         # Use a default distribution if one is not set
#         if self.distribution is None:
#             if env_params.action_continuous:
#                 self.distribution = dist.Normal
#             else:
#                 self.distribution = dist.Categorical

#         # Set the correct action dimension for the network
#         if env_params.action_continuous:
#             final_act = None
#         else:
#             final_act = torch.nn.Softmax(dim=-1)

#         # Initialize Actor and Critic Networks
#         self.action_dim = self.env_params.action_len
#         self.num_dist_params = self.distribution.parameters_per_action()
#         self.actor_network = MLP(
#             state_dim=self.env_params.observation_shape[0],
#             action_dim=self.action_dim * self.num_dist_params,
#             final_activation=final_act,
#         )
#         self.critic_network = MLP(
#             state_dim=self.env_params.observation_shape[0],
#             action_dim=1,
#         )

#         self.current_dist = None
#         self.value_estimates = None
#         self.action_log_probs = None
#         self.entropy = None

#     def get_actor_network(self):
#         return self.actor_network

#     def get_critic_network(self):
#         return self.critic_network

#     def get_action(self, state: TensorDict) -> TensorDict:
#         obs = state['observation']
#         action_probs = self.actor_network(obs)
#         action_probs = action_probs.view(-1, self.action_dim, self.num_dist_params)
#         dist = self.distribution(action_probs)
#         action = dist.sample()

#         # @todo clean this up in the distribution interface
#         if len(action.shape) == 1:
#             action = action.unsqueeze(-1)

#         self.current_dist = dist
#         self.value_estimates = self.critic_network(obs)
#         self.entropy = dist.entropy()

#         state['action'] = action
#         return state

#     def get_value_estimates(self) -> torch.Tensor:
#         return self.value_estimates

#     def get_log_probs(self, actions) -> torch.Tensor:
#         log_probs = self.current_dist.log_prob(actions.squeeze())
#         return log_probs.unsqueeze(-1)

#     def get_entropy(self) -> torch.Tensor:
#         return self.entropy

# class TabularPolicy(BasePolicy):
#     """
#     Base class for implementing tabular policies.

#     Args:
#         env_params (EnvParams): Environment parameters.
#         device (str): The device to use.
#     """
#     def __init__(self,
#                  env_params: EnvParams,
#                  q_table: Optional[Any] = None,
#                  decision_function: Optional[DecisionFunction] = EpsilonGreedy(),
#                  device: str = 'cpu',
#                  ) -> None:
#         super().__init__(env_params, device)
