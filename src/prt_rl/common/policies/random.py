"""
Random Policy that samples actions uniformly from the action space.
"""
import torch
from typing import Union
from prt_rl.env.interface import EnvParams, MultiAgentEnvParams

class RandomPolicy:
    """
    Implements a policy that uniformly samples random actions.

    This policy implements the Policy protocol so it can be used with any Collector or Evaluator in the PRT-RL framework. 
    
    Args:
        env_params (EnvParams): environment parameters
    """
    def __init__(self,
                 env_params: Union[EnvParams | MultiAgentEnvParams],
                 ) -> None:
        self.env_params = env_params

    @torch.no_grad()
    def act(self,
                   obs: torch.Tensor,
                   deterministic: bool = False
                   ) -> torch.Tensor:
        """
        Randomly samples an action from action space.

        Returns:
            TensorDict: Tensordict with the "action" key added
        """
        if deterministic:
            raise ValueError("RandomPolicy does not support deterministic actions. Set deterministic=False to sample random actions.")
        
        if isinstance(self.env_params, EnvParams):
            ashape = (obs.shape[0], self.env_params.action_len)
            params = self.env_params
        elif isinstance(self.env_params, MultiAgentEnvParams):
            ashape = (obs.shape[0], self.env_params.num_agents, self.env_params.agent.action_len)
            params = self.env_params.agent
        else:
            raise ValueError("env_params must be a EnvParams or MultiAgentEnvParams")

        if not params.action_continuous:
            # Add 1 to the high value because randint samples between low and 1 less than the high: [low,high)
            action = torch.randint(low=params.action_min, high=params.action_max + 1,
                                   size=ashape)
        else:
            action = torch.rand(size=ashape)

            # Scale the random [0,1] actions to the action space [min,max]
            max_actions = torch.tensor(params.action_max).unsqueeze(0)
            min_actions = torch.tensor(params.action_min).unsqueeze(0)
            action = action * (max_actions - min_actions) + min_actions

        return action
        