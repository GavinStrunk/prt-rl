from tensordict import TensorDict
import torch
from typing import Optional
from prt_rl.env.interface import EnvParams
from prt_rl.common.policy.policies import Policy
from prt_rl.common.networks import MLP
import prt_rl.common.distributions as dist

class ActorCriticPolicy(Policy):
    """
    Actor critic policy
    """
    def __init__(self,
                 env_params: EnvParams,
                 distribution: Optional[dist.Distribution] = None,
                 device: str = "cpu",
                 ) -> None:
        super().__init__(env_params=env_params, device=device)
        self.distribution = distribution

        # Use a default distribution if one is not set
        if self.distribution is None:
            if env_params.action_continuous:
                self.distribution = dist.Normal
            else:
                self.distribution = dist.Categorical

        # Set the correct action dimension for the network
        if env_params.action_continuous:
            final_act = None
        else:
            final_act = torch.nn.Softmax(dim=-1)

        # Initialize Actor and Critic Networks
        self.action_dim = self.env_params.action_len
        self.num_dist_params = self.distribution.parameters_per_action()
        self.actor_network = MLP(
            state_dim=self.env_params.observation_shape[0],
            action_dim=self.action_dim * self.num_dist_params,
            final_activation=final_act,
        )
        self.critic_network = MLP(
            state_dim=self.env_params.observation_shape[0],
            action_dim=1,
        )

        self.current_dist = None
        self.value_estimates = None
        self.action_log_probs = None
        self.entropy = None

    def get_actor_network(self):
        return self.actor_network

    def get_critic_network(self):
        return self.critic_network

    def get_action(self, state: TensorDict) -> TensorDict:
        obs = state['observation']
        action_probs = self.actor_network(obs)
        action_probs = action_probs.view(-1, self.action_dim, self.num_dist_params)
        dist = self.distribution(action_probs)
        action = dist.sample()

        # @todo clean this up in the distribution interface
        if len(action.shape) == 1:
            action = action.unsqueeze(-1)

        self.current_dist = dist
        self.value_estimates = self.critic_network(obs)
        self.entropy = dist.entropy()

        state['action'] = action
        return state

    def get_value_estimates(self) -> torch.Tensor:
        return self.value_estimates

    def get_log_probs(self, actions) -> torch.Tensor:
        log_probs = self.current_dist.log_prob(actions.squeeze())
        return log_probs.unsqueeze(-1)

    def get_entropy(self) -> torch.Tensor:
        return self.entropy
