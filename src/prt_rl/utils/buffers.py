from abc import ABC, abstractmethod
from collections import deque
import random
from tensordict.tensordict import TensorDict

class BaseReplayBuffer(ABC):
    def __init__(self,
                 capacity: int,
                 device: str = 'cpu'
                 ) -> None:
        self.capacity = capacity
        self.device = device

    @abstractmethod
    def add(self, experience: TensorDict) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int) -> TensorDict:
        raise NotImplementedError


class ReplayBuffer(BaseReplayBuffer):
    def __init__(self,
                 capacity: int
                 ) -> None:
        super().__init__(capacity)
        self.memory = deque(maxlen=self.capacity)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        return experiences

    def __len__(self):
        return len(self.memory)