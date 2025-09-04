from dataclasses import dataclass
import torch
from typing import Optional, List, Any
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.env.interface import EnvironmentInterface, EnvParams
from prt_rl.common.evaluators import Evaluator
from prt_rl.agent import BaseAgent

from prt_rl.model_based.planners.cross_entropy import CrossEntropyMethodPlanner
from prt_rl.model_based.planners.rollout import rollout_action_sequence


@dataclass
class CEMConfig:
    """
    Configuration parameters for the Cross-Entropy Method (CEM) agent.

    The number of elites is typically around 10% of the number of action sequences.

    Attributes:
        planning_horizon (int): Number of steps to plan ahead.
        num_iterations (int): Number of CEM iterations to perform.
        num_samples (int): Number of action sequences to sample per iteration.
        num_elites (int): Number of top-performing sequences to use for updating the distribution.
        initial_std (float): Initial standard deviation for the Gaussian sampling distribution.
    """
    num_action_sequences: int = 100
    planning_horizon: int = 10
    num_iterations: int = 5
    num_elites: int = 10
    use_smoothing: bool = False
    use_clipping: bool = False
    tau: Optional[float] = None
    beta: float = 0.2

class CEM(BaseAgent):
    """
    Cross-Entropy Method (CEM) Agent for Model-Based Reinforcement Learning.
    
    This agent uses the CEM algorithm to select actions based on either a known or learned model of the environment.
    """
    def __init__(self,
                 env_params: EnvParams,
                 model_config: Any,
                 model_fcn: callable,
                 reward_fcn: callable,
                 config: CEMConfig = CEMConfig(),
                 device: str = 'cpu'
                 ) -> None:
        super().__init__()
        self.env_params = env_params
        self.model_config = model_config
        self.model_fcn = model_fcn
        self.reward_fcn = reward_fcn
        self.config = config
        self.device = device

        self.planner = CrossEntropyMethodPlanner(
            action_mins=env_params.get_action_min_tensor(),
            action_maxs=env_params.get_action_max_tensor(),
            **self.config.__dict__,
            device=device
        )

    def predict(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Plan action sequences using the CEM algorithm and return the first action of the best sequence.
        
        Args:
            state (torch.Tensor): The current state of the environment. Shape (state_dim,).
            deterministic (bool): If True, always return the same action for the same state.
            
        Returns:
            torch.Tensor: The selected action. Shape (action_dim,).
        """
        return self.planner.plan(self.model_fcn, self.model_config, self.reward_fcn, state)

    def train(self,
              env: 'EnvironmentInterface',
              total_steps: int,
              schedulers: 'Optional[List[ParameterScheduler]]' = None,
              logger: 'Optional[Logger]' = None,
              evaluator: 'Optional[Evaluator]' = None,
              show_progress: bool = True
              ) -> None:
        raise NotImplementedError("The train method is not implemented for Cross Entropy Method agent.")