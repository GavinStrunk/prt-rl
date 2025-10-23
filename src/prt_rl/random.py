"""
Random Agent that samples actions uniformly from the action space.
"""
import torch
from typing import Optional, List, Union
from prt_rl.agent import BaseAgent
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.env.interface import EnvironmentInterface, EnvParams, MultiAgentEnvParams
from prt_rl.common.evaluators import Evaluator

class RandomAgent(BaseAgent):
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

    def predict(self,
                   state: torch.Tensor,
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
            ashape = (state.shape[0], self.env_params.action_len)
            params = self.env_params
        elif isinstance(self.env_params, MultiAgentEnvParams):
            ashape = (state.shape[0], self.env_params.num_agents, self.env_params.agent.action_len)
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
    
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              evaluator: Optional[Evaluator] = None,
              show_progress: bool = True
              ) -> None:
        raise NotImplementedError("RandomAgent does not support training. It is designed for interactive control only.")