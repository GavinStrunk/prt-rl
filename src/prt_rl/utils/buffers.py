import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.index = 0
        self.size = 0

        self.states = torch.zeros((capacity, state_dim))
        self.actions = torch.zeros((capacity, action_dim))
        self.rewards = torch.zeros((capacity, 1))
        self.dones = torch.zeros((capacity, 1))
        self.next_states = torch.zeros((capacity, state_dim))

    def add(self, state, action, reward, done, next_state):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.next_states[self.index] = next_state

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, replace=False):
        indices = torch.randint(0, self.size, (batch_size,))
        return {
            "state": self.states[indices],
            "action": self.actions[indices],
            "reward": self.rewards[indices],
            "done": self.dones[indices],
            "next_state": self.next_states[indices],
        }