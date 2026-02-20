"""
Agent Interface for implementing new agents.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
import torch
from typing import Optional, Union, List
from prt_rl.common.schedulers import ParameterScheduler
import prt_rl.common.utils as utils


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
    
    # --------------
    # Helper Methods
    # --------------
    @classmethod
    def _update_schedulers(
        cls,
        schedulers: Optional[List[ParameterScheduler]] = None,
        step: int = 0
    ) -> None:
        """
        Update a list of parameter schedulers to the current step.

        Args:
            schedulers (Optional[List[ParameterScheduler]]): List of schedulers to update. Each scheduler should have an update(current_step: int) method. Default is None.
            step (int): The current step to update the schedulers to. Scalar.
        Returns:
            None
        """
        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.update(current_step=step)

    @classmethod
    def _update_optimizer(
        cls,
        optimizer: object,
        learning_rate: float
    ) -> None:
        """
        Update the learning rate for all parameter groups in an optimizer.

        Args:
            optimizer (object): Optimizer object (e.g., torch.optim.Optimizer) with a param_groups attribute (list of dicts with 'lr' key).
            learning_rate (float): New learning rate to set. Scalar.
        Returns:
            None
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    @classmethod
    def _compute_gae(
        cls,
        experience: dict,
        gamma: float,
        gae_lambda: float
    ) -> dict:
        """
        Compute Generalized Advantage Estimation (GAE) and returns, and flatten the experience batch.

        Args:
            experience (dict): Dictionary with keys:
                'reward' (torch.Tensor): Rewards, shape (T, N, 1) or (B, 1)
                'value' (torch.Tensor): State values, shape (T, N, 1) or (B, 1)
                'done' (torch.Tensor): Done flags, shape (T, N, 1) or (B, 1)
                'last_value_est' (torch.Tensor): Value estimates for final state, shape (N, 1)
            gamma (float): Discount factor. Scalar.
            gae_lambda (float): GAE lambda. Scalar.
        Returns:
            dict: Experience dict with added keys:
                'advantages' (torch.Tensor): Estimated advantages, shape (N*T, ...)
                'returns' (torch.Tensor): TD(lambda) returns, shape (N*T, ...)
                All other tensors are flattened to (N*T, ...). 'last_value_est' is removed.
        """
        # Compute Advantages and Returns under the current policy
        advantages, returns = utils.generalized_advantage_estimates(
            rewards=experience['reward'],
            values=experience['value'],
            dones=experience['done'],
            last_values=experience['last_value_est'],
            gamma=gamma,
            gae_lambda=gae_lambda
        )

        experience['advantages'] = advantages.detach()
        experience['returns'] = returns.detach()

        # Flatten the experience batch (N, T, ...) -> (N*T, ...) and remove the last_value_est key because we don't need it anymore
        experience = {k: v.reshape(-1, *v.shape[2:]) for k, v in experience.items() if k != 'last_value_est'}

        return experience
