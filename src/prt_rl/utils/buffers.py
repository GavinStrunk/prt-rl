from abc import ABC, abstractmethod
import torch
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
    """
    Replay buffer stores experience tuples from environment interactions.
    
    """
    def __init__(self,
                 capacity: int,
                 state_dim: int,
                 action_dim: int,
                 device: str = 'cpu'
                 ) -> None:
        super().__init__(capacity, device)
        self.index = 0
        self.size = 0

        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)
        self.next_states = torch.zeros((capacity, state_dim))

    def add(self,
            experience: TensorDict
            ) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.next_states[self.index] = next_state

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self,
               batch_size: int
               ) -> TensorDict:
        indices = torch.randint(0, self.size, (batch_size,))
        return {
            "state": self.states[indices],
            "action": self.actions[indices],
            "reward": self.rewards[indices],
            "done": self.dones[indices],
            "next_state": self.next_states[indices],
        }