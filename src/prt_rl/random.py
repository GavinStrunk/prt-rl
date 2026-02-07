"""
Random Agent that samples actions uniformly from the action space.
"""
from pathlib import Path
import torch
from typing import Union
from prt_rl.agent import AgentInterface
from prt_rl.env.interface import EnvParams, MultiAgentEnvParams

class RandomAgent(AgentInterface):
    """
    Implements a policy that uniformly samples random actions.

    Args:
        env_params (EnvParams): environment parameters
    """
    def __init__(self,
                 env_params: Union[EnvParams | MultiAgentEnvParams],
                 ) -> None:
        super(RandomAgent, self).__init__()
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
            raise ValueError("RandomAgent does not support deterministic actions. Set deterministic=False to sample random actions.")
        
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
    
    def _save_impl(self, path: Union[str, Path]) -> None:
        pass

    def load(cls, path: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> "AgentInterface":
        pass
    