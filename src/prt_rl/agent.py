"""
Agent Interface for implementing new agents.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
import torch
from typing import Optional, Union


class Agent(ABC):
    """
    Base class for all agents in the PRT-RL framework.
    """
    def __init__(self, 
                 device: str = "cpu"
                 ) -> None:
        self.device = torch.device(device)

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
        raise NotImplementedError("The act method must be implemented by subclasses.")

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

    @abstractmethod
    def _save_impl(self, path: Path) -> None:
        raise NotImplementedError("The _save_impl method must be implemented by subclasses.")

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "Agent":
        raise NotImplementedError("The load method must be implemented by subclasses.")
    
    
