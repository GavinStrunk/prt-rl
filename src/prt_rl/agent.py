"""
Base Agent Interface for implementing new agents.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
import torch
from typing import Optional, List, Union, Tuple
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.policies as pmod


class BaseAgent(ABC):
    """
    Base class for all agents in the PRT-RL framework.
    """
    def __init__(self, 
                 policy: pmod.PolicyModule, 
                 device: str = "cpu"
                 ) -> None:
        self.policy = policy
        self.device = torch.device(device)
        self.policy.to(self.device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Perform an action based on the current state.

        Args:
            obs (torch.Tensor): The current observation from the environment.
            deterministic (bool): If True, the agent will select actions deterministically.

        Returns:
            torch.Tensor: The action to be taken.
        """
        raise self.policy.act(obs.to(self.device), deterministic=deterministic)[0]

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        if path is None:
            tmp = tempfile.TemporaryDirectory(prefix="prt_rl_ckpt_")
            path = Path(tmp.name)
            # keep reference so directory is not deleted immediately
            self._last_tmp_checkpoint = tmp
        else:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

        self._save_impl(path)
        return path

    def _save_impl(self, path: Path) -> None:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "BaseAgent":
        raise NotImplementedError("The load method must be implemented by subclasses.")
    
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
    

