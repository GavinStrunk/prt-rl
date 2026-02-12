"""
Base class for implementing policy modules.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import torch
from torch import Tensor
from typing import Tuple, Optional, Dict, Union

InfoDict = Dict[str, Tensor]

class Policy(torch.nn.Module, ABC):
    """
    Minimal runtime policy API used by collectors.

    - Must be an nn.Module (so .to(), .parameters(), etc.)
    - act() returns (action, info_dict). info_dict can include "log_prob", "value", etc.
    - reset() is optional for RNN state.
    """
    @abstractmethod
    def act(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, InfoDict]:
        """
        Given an observation, return an action and an info dictionary. Commone keys in the info dictionary include "log_prob" and "value".
        
        Args:
            obs (Tensor): The observation tensor.
            deterministic (bool): Whether to use deterministic actions.
        Returns:
            Tuple[Tensor, InfoDict]: A tuple containing the action tensor and an info dictionary.
        """
        raise NotImplementedError

    def reset(self, batch_size: Optional[int] = None) -> None:
        """
        Reset any internal state (e.g., RNN hidden states). If batch_size is provided, reset for that batch size.
        
        Args:
            batch_size (Optional[int]): The batch size for resetting internal states.
        """
        return

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the module's parameters or buffers are located.
        
        Returns:
            torch.device: The device of the module.
        """
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    def save(self, path: Union[str, Path]) -> None:
        """
        Save a fully-constructed policy module.

        Note:
            This uses PyTorch module pickling, so loading requires the policy
            class to be importable in the runtime.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, Path(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        map_location: str | torch.device = "cpu",
    ) -> "Policy":
        """
        Load a policy saved with `Policy.save`.
        """
        policy = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(policy, cls):
            raise TypeError(f"Loaded policy type {type(policy)} is not an instance of {cls}.")
        return policy.to(map_location)
