from abc import ABC, abstractmethod
import torch
from typing import Dict


class BaseReplayBuffer(ABC):
    def __init__(self,
                 capacity: int,
                 device: str = 'cpu'
                 ) -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self.size = 0
        self.pos = 0

    def get_size(self) -> int:
        """
        Returns the current number of elements in the replay buffer.
        Returns:
            int: The current size of the replay buffer.
        """
        return self.size
    
    def __len__(self) -> int:
        """
        Returns the current number of elements in the replay buffer.
        Returns:
            int: The current size of the replay buffer.
        """
        return self.size

    @abstractmethod
    def add(self, experience: Dict[str, torch.Tensor]) -> None:
        """
        Adds a new experience to the replay buffer.
        Args:
            experience (Dict[str, torch.Tensor]): A dictionary containing the experience data.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of experiences from the replay buffer.
        Args:
            batch_size (int): The number of samples to draw from the buffer.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the sampled experiences.
        """
        raise NotImplementedError
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clears the replay buffer, resetting its size and position.
        """
        raise NotImplementedError


class ReplayBuffer(BaseReplayBuffer):
    """
    A circular replay buffer that overwrites old experiences when full.
    
    Args:
        capacity (int): The maximum number of experiences to store.
        device (torch.device): The device to store the buffer on (default: CPU).
    """
    def __init__(self, capacity: int, device: torch.device = torch.device("cpu")):
        super().__init__(capacity, device)
        self.buffer = {}
        self.initialized = False

    def _init_storage(self, transition: Dict[str, torch.Tensor]) -> None:
        """
        Initializes the storage for the replay buffer based on the first transition.
        
        Args:
            transition (Dict[str, torch.Tensor]): A dictionary containing the transition data.
        """
        for k, v in transition.items():
            shape = (self.capacity,) + v.shape[1:]  # Skip batch dim
            self.buffer[k] = torch.zeros(shape, dtype=v.dtype, device=self.device)
        self.initialized = True

    def add(self, 
            transition: Dict[str, torch.Tensor]
            ) -> None:
        """
        Adds a new transition to the replay buffer.

        Args:
            transition (Dict[str, torch.Tensor]): A dictionary containing the transition data.
        """
        if not self.initialized:
            self._init_storage(transition)

        batch_size = next(iter(transition.values())).shape[0]
        insert_end = self.pos + batch_size

        if insert_end <= self.capacity:
            # One contiguous block
            idx = slice(self.pos, insert_end)
            for k, v in transition.items():
                self.buffer[k][idx] = v.to(self.device)
        else:
            # Wrap-around: split into two writes
            first_len = self.capacity - self.pos
            second_len = batch_size - first_len
            for k, v in transition.items():
                self.buffer[k][self.pos:] = v[:first_len].to(self.device)
                self.buffer[k][:second_len] = v[first_len:].to(self.device)

        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of transitions from the replay buffer.
        Args:
            batch_size (int): The number of samples to draw from the buffer.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the sampled transitions.
        """
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer to sample.")

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {k: v[indices] for k, v in self.buffer.items()}
    
    def clear(self) -> None:
        """
        Clears the replay buffer, resetting its state.
        """
        self.size = 0
        self.pos = 0
        self.buffer = {}
        self.initialized = False