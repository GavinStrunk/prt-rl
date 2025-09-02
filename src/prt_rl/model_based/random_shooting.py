from dataclasses import dataclass
import torch
from typing import Optional, List, Any
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.env.interface import EnvironmentInterface, EnvParams
from prt_rl.common.evaluators import Evaluator
from prt_rl.agent import BaseAgent

from prt_rl.model_based.planners.shooting import RandomShootingPlanner
from prt_rl.model_based.planners.rollout import rollout_action_sequence

@dataclass
class RandomShootingConfig:
    """
    Configuration parameters for the Random Shooting agent.

    Attributes:
        planning_horizon (int): Number of steps to plan ahead.
        num_action_sequences (int): Number of random action sequences to sample.
        num_elites (int): Number of top action sequences to consider for selecting the best action.
    """
    planning_horizon: int = 10
    num_action_sequences: int = 100 

class RandomShooting(BaseAgent):
    """
    Random Shooting Agent for Model-Based Reinforcement Learning.
    
    This agent uses a random shooting method to select actions based on either a known or learned model of the environment.
    """
    def __init__(self,
                 env_params: EnvParams,
                 model_config: Any,
                 model_fcn: callable,
                 reward_fcn: callable,
                 config: RandomShootingConfig = RandomShootingConfig(),
                 device: str = 'cpu'
                 ) -> None:
        super().__init__()
        self.env_params = env_params
        self.model_config = model_config
        self.model_fcn = model_fcn
        self.reward_fcn = reward_fcn
        self.config = config
        self.device = device

        self.planner = RandomShootingPlanner(
            action_mins=env_params.get_action_min_tensor(),
            action_maxs=env_params.get_action_max_tensor(),
            planning_horizon=config.planning_horizon,
            device=device
        )

    def predict(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Plan action sequences
        num_action_seq = self.config.num_action_sequences

        # Random plan action sequences with shape (B, H, action_dim)
        action_sequences = self.planner.plan(num_action_seq)
        
        # Evaluate action sequences using the model and reward function
        rollout = rollout_action_sequence(self.model_config, self.model_fcn, state, action_sequences)
        
        # Evaluate the objective value (reward) for each action sequence
        rewards = self.reward_fcn(rollout['state'], rollout['action'], rollout['next_state']) 

        # Sort by reward value
        max_index = torch.argmax(rewards)

        # Return the first action from the best action sequence
        return action_sequences[max_index.item(), 0, :].unsqueeze(0)

    def train(self,
              env: 'EnvironmentInterface',
              total_steps: int,
              schedulers: 'Optional[List[ParameterScheduler]]' = None,
              logger: 'Optional[Logger]' = None,
              evaluator: 'Optional[Evaluator]' = None,
              show_progress: bool = True
              ) -> None:
        raise NotImplementedError("The train method is not implemented for Random Shooting agent.")