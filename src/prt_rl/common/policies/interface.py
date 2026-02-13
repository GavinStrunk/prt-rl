"""Policy interfaces used across algorithms."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple, Union, runtime_checkable, TypeVar, Type, Any

import torch
from torch import Tensor
from prt_rl.common.decision_functions import DecisionFunction

T = TypeVar("T", bound="TabularPolicy")

@runtime_checkable
class Policy(Protocol):
    """Runtime acting interface consumed by collectors."""

    def act(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return an action tensor and auxiliary policy outputs."""
        ...

class NeuralPolicy(torch.nn.Module, ABC):
    """
    Base class for torch-backed policies.

    Implements the Policy protocol and adds utility methods for saving/loading and device management.
    """

    @abstractmethod
    def act(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return an action tensor and auxiliary policy outputs."""
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, Path(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        map_location: str | torch.device = "cpu",
    ) -> "NeuralPolicy":
        policy = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(policy, cls):
            raise TypeError(f"Loaded policy type {type(policy)} is not an instance of {cls}.")
        return policy.to(map_location)


class TabularPolicy(ABC):
    """Base class for tabular policies (non-Module)."""
    def __init__(self, table: Tensor, decision_function: DecisionFunction) -> None:
        self.table = table
        self.decision_function = decision_function

    @abstractmethod
    def act(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return an action tensor and auxiliary policy outputs."""
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self.table.device

    @property
    def dtype(self) -> torch.dtype:
        return self.table.dtype

    def to(self: T, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> T:
        self.table = self.table.to(device=device, dtype=dtype if dtype is not None else self.table.dtype)
        return self

    def clone(self: T) -> T:
        # create a new instance of the same class with a cloned table
        return type(self).from_snapshot(self.snapshot().copy())  # uses your snapshot contract

    # ---- Serialization (non-Module naming) ----
    def snapshot(self) -> Dict[str, Any]:
        """
        Return a serializable snapshot.

        Subclasses can extend this by `snap = super().snapshot(); snap[...] = ...`.
        """
        return {
            "type": type(self).__name__,
            "format_version": 1,
            "table": self.table,
            "decision_function": self.decision_function.to_dict(),
        }

    @classmethod
    def from_snapshot(cls: Type[T], snapshot: Dict[str, Any]) -> T:
        # Base restores only `table`. Subclasses can override if they add fields.
        table = snapshot["table"]
        decision_function = DecisionFunction.from_dict(snapshot["decision_function"])
        return cls(table=table, decision_function=decision_function)  # type: ignore[arg-type]

    def save(self, path: str) -> None:
        torch.save(self.snapshot(), path)

    @classmethod
    def load(cls: Type[T], path: str, map_location: Optional[torch.device | str] = None) -> T:
        snap = torch.load(path, map_location=map_location)
        # If you want, enforce type match here.
        return cls.from_snapshot(snap)