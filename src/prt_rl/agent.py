from abc import ABC, abstractmethod
import torch
from typing import Optional, List, Union
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.common.policies import BasePolicy
from prt_rl.env.interface import EnvironmentInterface, EnvParams, MultiAgentEnvParams
from prt_rl.common.evaluators import Evaluator

class BaseAgent(ABC):
    """
    Base class for all agents in the PRT-RL framework.
    """
    def __init__(self,
                 policy: BasePolicy
                 ) -> None:
        self.policy = policy

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """
        Call the agent to perform an action based on the current state.

        Args:
            state: The current state of the environment.

        Returns:
            The action to be taken.
        """
        return self.predict(state)

    @abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform an action based on the current state.

        Args:
            state: The current state of the environment.

        Returns:
            The action to be taken.
        """
        raise NotImplementedError("The predict method must be implemented by subclasses.")

    @abstractmethod
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              logging_freq: int = 1000,
              evaluator: Evaluator = Evaluator(),
              eval_freq: int = 1000,
              ) -> None:
        """
        Update the agent's knowledge based on the action taken and the received reward.

        Args:
            state: The current state of the environment.
            action: The action taken.
            reward: The reward received after taking the action.
            next_state: The next state of the environment after taking the action.
        """
        raise NotImplementedError("The train method must be implemented by subclasses.")
    

class RandomAgent(BaseAgent):
    """
    Implements a policy that uniformly samples random actions.

    Args:
        env_params (EnvParams): environment parameters
    """
    def __init__(self,
                 env_params: Union[EnvParams | MultiAgentEnvParams],
                 ) -> None:
        super(RandomAgent, self).__init__(policy=None)
        self.env_params = env_params

    def predict(self,
                   state: torch.Tensor
                   ) -> torch.Tensor:
        """
        Randomly samples an action from action space.

        Returns:
            TensorDict: Tensordict with the "action" key added
        """
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
    
    def train(self, env, total_steps, schedulers = None, logger = None, logging_freq = 1000):
        raise NotImplementedError("RandomAgent does not support training. It is designed for interactive control only.")