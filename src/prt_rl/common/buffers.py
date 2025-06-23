from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict, Tuple


class BaseBuffer(ABC):
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


class ReplayBuffer(BaseBuffer):
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

    def _init_storage(self, experience: Dict[str, torch.Tensor]) -> None:
        """
        Initializes the storage for the replay buffer based on the first transition.
        
        Args:
            experience (Dict[str, torch.Tensor]): A dictionary containing the transition data.
        """
        for k, v in experience.items():
            shape = (self.capacity,) + v.shape[1:]  # Skip batch dim
            self.buffer[k] = torch.zeros(shape, dtype=v.dtype, device=self.device)
        self.initialized = True

    def add(self, 
            experience: Dict[str, torch.Tensor]
            ) -> None:
        """
        Adds a new transition to the replay buffer.

        Args:
            experience (Dict[str, torch.Tensor]): A dictionary containing the transition data.
        """
        if not self.initialized:
            self._init_storage(experience)

        batch_size = next(iter(experience.values())).shape[0]
        insert_end = self.pos + batch_size

        if insert_end <= self.capacity:
            # One contiguous block
            idx = slice(self.pos, insert_end)
            for k, v in experience.items():
                self.buffer[k][idx] = v.to(self.device)
        else:
            # Wrap-around: split into two writes
            first_len = self.capacity - self.pos
            second_len = batch_size - first_len
            for k, v in experience.items():
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
    
    def resize(self, new_capacity: int):
        """
        Expands the buffer to a new capacity while preserving existing data.
        Args:
            new_capacity (int): The new buffer capacity.
        """
        if new_capacity <= self.capacity:
            raise ValueError("New capacity must be greater than current capacity.")

        new_buffer = {}
        for k, v in self.buffer.items():
            new_shape = (new_capacity,) + v.shape[1:]
            new_tensor = torch.zeros(new_shape, dtype=v.dtype, device=self.device)
            if self.pos >= self.size:
                # No wrap-around
                new_tensor[:self.size] = v[:self.size]
            else:
                # Wrap-around logic
                new_tensor[:self.capacity - self.pos] = v[self.pos:]
                new_tensor[self.capacity - self.pos:self.size] = v[:self.pos]
            new_buffer[k] = new_tensor

        self.buffer = new_buffer
        self.capacity = new_capacity
        self.pos = self.size  # Next write after last element    

    def get_batches(self, batch_size: int):
        """
        Yields shuffled mini-batches from the buffer.
        """
        if self.size == 0:
            return
        indices = torch.randperm(self.size, device=self.device)
        for i in range(0, self.size, batch_size):
            idx = indices[i:i + batch_size]
            yield {k: v[idx] for k, v in self.buffer.items()}        
    
    def clear(self) -> None:
        """
        Clears the replay buffer, resetting its state.
        """
        self.size = 0
        self.pos = 0
        self.buffer = {}
        self.initialized = False

    def save(self, path: str) -> None:
        """
        Saves the replay buffer to a file.

        Args:
            path (str): Path to the file where the buffer will be saved.
        """
        torch.save({
            'buffer': self.buffer,
            'size': self.size,
            'pos': self.pos,
            'capacity': self.capacity
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ReplayBuffer":
        """
        Loads a replay buffer from a file.

        Args:
            path (str): Path to the saved buffer file.
            device (str): Device to load the buffer to. Defaults to "cpu".

        Returns:
            ReplayBuffer: A ReplayBuffer instance with restored data.
        """
        data = torch.load(path, map_location=torch.device(device))

        obj = cls(capacity=data['capacity'], device=torch.device(device))
        obj.buffer = {k: v.to(torch.device(device)) for k, v in data['buffer'].items()}
        obj.size = data['size']
        obj.pos = data['pos']
        obj.initialized = True
        return obj        

class SumTree:
    """
    A binary sum tree for efficient sampling of elements proportional to their priority.

    The SumTree is a binary tree where each parent node stores the sum of its child nodes. It's particularly useful for:  

    - Efficient sampling from a discrete probability distribution:  
        - You can sample an index i proportional to a weight p_i in O(log N) time.
    - Efficient dynamic updates of weights (e.g., priority values) while maintaining cumulative structure.

    **Key Applications:**  

    - Prioritized Experience Replay (PER): Sample transitions with probability ∝ priority.
    - Importance Sampling in any algorithm that requires sampling proportional to some non-uniform, changeable weights.
    - Event scheduling in simulations: Where some events happen more frequently than others.

    Attributes:
        capacity (int): Maximum number of elements the tree can hold.
        tree (np.ndarray): Binary tree storing priorities.
        data_pointer (int): Current position to insert new priority.
    """
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0

    def add(self, priority: float) -> None:
        """
        Add a new priority to the sum tree.

        Args:
            priority (float): Priority value to insert.
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx: int, priority: float) -> None:
        """
        Update the priority at a specific tree index and propagate the change.

        Args:
            tree_idx (int): Index in the tree to update.
            priority (float): New priority value.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value: float) -> Tuple[int, float, int]:
        """
        Traverse the tree to find the leaf node corresponding to the sample value.

        Args:
            value (float): A sample value in [0, total_priority).

        Returns:
            Tuple[int, float, int]: (tree index, priority, data index)
        """
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if value <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                value -= self.tree[left_idx]
                parent_idx = right_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_priority(self) -> float:
        """
        Returns the sum of all priorities.

        Returns:
            float: Total priority.
        """
        return self.tree[0]


class PrioritizedReplayBuffer(BaseBuffer):
    """
    A Prioritized Experience Replay Buffer using a SumTree for efficient sampling.

    Attributes:
        alpha (float): How much prioritization is used (0 = uniform, 1 = full prioritization).
        beta (float): Importance sampling bias correction term.
        priorities (SumTree): Sum tree to manage priorities.
        max_priority (float): The maximum priority value observed.
    """
    def __init__(self,
                 capacity: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 device: str = 'cpu'
                 ) -> None:
        super().__init__(capacity, device)
        self.alpha = alpha
        self.beta = beta
        self.beta0 = beta
        self.priorities = SumTree(capacity)
        self.max_priority = 1.0
        self.buffer = {}
        self.initialized = False

    def _init_storage(self, experience: Dict[str, torch.Tensor]) -> None:
        """
        Initializes the storage for the replay buffer based on the first transition.
        Args:
            experience (Dict[str, torch.Tensor]): A dictionary containing the transition data.
        """
        for k, v in experience.items():
            shape = (self.capacity,) + v.shape[1:]
            self.buffer[k] = torch.zeros(shape, dtype=v.dtype, device=self.device)
        self.initialized = True

    def add(self, experience: Dict[str, torch.Tensor]) -> None:
        """
        Adds a new transition to the replay buffer.
        Args:
            experience (Dict[str, torch.Tensor]): A dictionary containing the transition data.
        """
        if not self.initialized:
            self._init_storage(experience)

        batch_size = next(iter(experience.values())).shape[0]
        insert_end = self.pos + batch_size

        if insert_end <= self.capacity:
            idx = slice(self.pos, insert_end)
            for k, v in experience.items():
                self.buffer[k][idx] = v.to(self.device)
        else:
            first_len = self.capacity - self.pos
            second_len = batch_size - first_len
            for k, v in experience.items():
                self.buffer[k][self.pos:] = v[:first_len].to(self.device)
                self.buffer[k][:second_len] = v[first_len:].to(self.device)

        for _ in range(batch_size):
            self.priorities.add(self.max_priority ** self.alpha)

        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of transitions from the replay buffer using prioritized sampling.
        Returns a dictionary containing: 
            - 'weights': Importance sampling weights for each sample.
            - 'indices': Indices of the sampled transitions in the buffer.
            - Other transition data (e.g., state, action, reward, etc.).
            
        Args:
            batch_size (int): The number of samples to draw from the buffer.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the sampled transitions.
        """
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer to sample.")

        segment = self.priorities.total_priority() / batch_size
        indices = []
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            leaf_idx, priority, data_idx = self.priorities.get_leaf(s)
            indices.append(data_idx)
            priorities.append(priority)

        indices_torch = torch.tensor(indices, dtype=torch.long, device=self.device)
        sampled = {k: v[indices_torch] for k, v in self.buffer.items()}

        priorities = torch.tensor(priorities, dtype=torch.float32, device=self.device)
        sampling_probabilities = priorities / self.priorities.total_priority()
        weights = (self.size * sampling_probabilities).pow(-self.beta)
        weights /= weights.max()

        sampled['weights'] = weights
        sampled['indices'] = indices_torch

        return sampled

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        """
        Update the priorities of the sampled transitions based on TD errors.

        Args:
            indices (torch.Tensor): Indices of the transitions to update.
            td_errors (torch.Tensor): TD errors for the transitions.
        """
        priorities = (td_errors.abs() + 1e-6).pow(self.alpha)
        for idx, priority in zip(indices.tolist(), priorities.tolist()):
            tree_idx = idx + self.capacity - 1
            self.priorities.update(tree_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def clear(self) -> None:
        """
        Clears the replay buffer, resetting its state.
        """
        self.size = 0
        self.pos = 0
        self.buffer = {}
        self.initialized = False
        self.priorities = SumTree(self.capacity)
        self.max_priority = 1.0
        self.beta = self.beta0

class RolloutBuffer(BaseBuffer):
    def __init__(self, 
                 capacity: int, 
                 device: str = 'cpu'
                 ) -> None:
        """
        Args:
            capacity: Max number of transitions the buffer can store.
            device: Torch device.
        """
        super().__init__(capacity, device)
        self.buffer: Dict[str, torch.Tensor] = {}
        self.initialized = False

    def _init_storage(self, experience: Dict[str, torch.Tensor]) -> None:
        for k, v in experience.items():
            shape = (self.capacity,) + v.shape[1:]
            self.buffer[k] = torch.zeros(shape, dtype=v.dtype, device=self.device)
        self.initialized = True

    def add(self, experience: Dict[str, torch.Tensor]) -> None:
        if not self.initialized:
            self._init_storage(experience)

        batch_size = next(iter(experience.values())).shape[0]
        if self.size + batch_size > self.capacity:
            raise ValueError("RolloutBuffer overflow: not enough capacity")

        idx = slice(self.size, self.size + batch_size)
        for k, v in experience.items():
            self.buffer[k][idx] = v.to(self.device)

        self.size += batch_size

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("Not enough samples to draw")

        indices = torch.randperm(self.size, device=self.device)[:batch_size]
        sampled = {k: v[indices] for k, v in self.buffer.items()}

        # Keep the remaining entries by copying them up
        keep_mask = torch.ones(self.size, dtype=bool, device=self.device)
        keep_mask[indices] = False
        keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)

        for k in self.buffer:
            self.buffer[k][:len(keep_indices)] = self.buffer[k][keep_indices]

        self.size -= batch_size
        return sampled

    def get_batches(self, batch_size: int):
        """
        Yields mini-batches in random order. The final batch may be smaller.

        After iteration, if drop_after_get=True, the buffer is cleared.
        """
        if self.size == 0:
            return

        indices = torch.randperm(self.size, device=self.device)

        for i in range(0, self.size, batch_size):
            idx = indices[i:i + batch_size]
            batch = {k: v[idx] for k, v in self.buffer.items()}
            yield batch

    def clear(self) -> None:
        self.size = 0
        self.buffer = {}
        self.initialized = False
