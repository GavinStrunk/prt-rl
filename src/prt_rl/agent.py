"""
Base Agent Interface for implementing new agents.
"""
from abc import ABC, abstractmethod
import torch
from typing import Optional, List
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.evaluators import Evaluator

class BaseAgent(ABC):
    """
    Base class for all agents in the PRT-RL framework.
    """
    def __call__(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Call the agent to perform an action based on the current state.

        Args:
            state (torch.Tensor): The current state of the environment.
            deterministic (bool): If True, the agent will select actions deterministically.

        Returns:
            torch.Tensor: The action to be taken.
        """
        return self.predict(state, deterministic)

    @abstractmethod
    def predict(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Perform an action based on the current state.

        Args:
            state (torch.Tensor): The current state of the environment.
            deterministic (bool): If True, the agent will select actions deterministically.

        Returns:
            torch.Tensor: The action to be taken.
        """
        raise NotImplementedError("The predict method must be implemented by subclasses.")

    @abstractmethod
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              evaluator: Optional[Evaluator] = None,
              show_progress: bool = True
              ) -> None:
        """
        Update the agent's knowledge based on the action taken and the received reward.

        Args:
            env (EnvironmentInterface): The environment in which the agent will operate.
            total_steps (int): Total number of training steps to perform.
            schedulers (List[ParameterScheduler]): List of parameter schedulers to update during training.
            logger (Optional[Logger]): Logger for logging training progress. If None, a default logger will be created.
            evaluator (Evaluator): Evaluator to evaluate the agent periodically.
            show_progress (bool): If True, show a progress bar during training.
        """
        raise NotImplementedError("The train method must be implemented by subclasses.")
    

