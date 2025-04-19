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

    @abstractmethod
    def add(self, experience: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int):
        raise NotImplementedError
    
    def __len__(self) -> int:
        return self.size


class ReplayBuffer(BaseReplayBuffer):
    def __init__(self,
                 capacity: int,
                 device: str = 'cpu'
                 ) -> None:
        super().__init__(capacity, device)
        self.buffer = {}
        self.initialized = False

    def _initialize_buffer(self, experience: Dict[str, torch.Tensor]) -> None:
        """
        Initializes the replay buffer with the given experience shape.

        Args:
            experience (Dict[str, torch.Tensor]): A dictionary containing the experience data,
        """
        for key, value in experience.items():
            self.buffer[key] = torch.zeros((self.capacity,) + value.shape, dtype=value.dtype, device=self.device)
        self.initialized = True

    def add(self, experience: Dict[str, torch.Tensor]) -> None:
        """
        Adds a new experience to the replay buffer.
        Args:
            experience (Dict[str, torch.Tensor]): A dictionary containing the experience data.
        """
        if not self.initialized:
            self._initialize_buffer(experience)

        
        num_envs = next(iter(experience.values())).shape[0]
        indices = (torch.arange(num_envs, device=self.device) + self.pos) % self.capacity

        for k, v in experience.items():
            if torch.device(v.device) != torch.device(self.device):
                raise RuntimeError(f"{k} is on {v.device}, but buffer expects {self.device}")
            self.buffer[k][indices] = v  # One batched insert

        self.pos = (self.pos + num_envs) % self.capacity
        self.size = min(self.size + num_envs, self.capacity)

    def sample(self, batch_size) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of experiences from the replay buffer.
        Args:
            batch_size (int): The number of samples to draw from the buffer.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the sampled experiences.
        """
        if self.size < batch_size:
            raise ValueError("Not enough samples to draw from the buffer")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {k: v[indices] for k, v in self.buffer.items()}
