import copy
from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
from typing import Optional

from tensordict import TensorDict
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.exact.trainers import TDTrainer
from prt_rl.utils.buffers import ReplayBuffer
from prt_rl.utils.decision_functions import DecisionFunction
from prt_rl.utils.policies import QNetworkPolicy

class DQN(TDTrainer):
    def __init__(self,
                 env: EnvironmentInterface,
                 num_envs: int = 1,
                 decision_function: Optional[DecisionFunction] = None,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 buffer_size: int = 50000,
                 min_buffer_size: int = 320,
                 mini_batch_size: int = 64,
                 target_update_steps: int = 15,
                 device: str = 'cpu'
                 ) -> None:
        self.env_params = env.get_parameters()
        self.num_envs = num_envs
        self.device = device
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=self.env_params.observation_shape,
            action_dim=self.env_params.action_shape,
            device=self.device
        )

        policy = QNetworkPolicy(
            env_params=self.env_params,
            num_envs=1,
            decision_function=decision_function,
            device=device
        )
        super(DQN, self).__init__(env, policy=policy)

        self.target_network = copy.deepcopy(self.policy)

    def update_policy(self, experience: TensorDict) -> None:
        pass


class DQN2:
    """
    Deep Q Network

    Args:
        state_size (int): size of state
        num_actions (int): number of actions
        alpha (float): learning rate
        gamma (float): discount factor
        buffer_size (int): replay buffer size

    """
    def __init__(self,
                 state_size: int,
                 num_actions: int,
                 alpha: float = 0.001,
                 epsilon: float = 0.2,
                 gamma: float = 0.9,
                 buffer_size: int = 100_000,
                 min_buffer_size: int = 320,
                 mini_batch_size: int = 64,
                 target_update_steps: int = 15
                 ) -> None:
        self.state_size = state_size
        self.num_actions = num_actions
        self.learning_rate = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.min_buffer_size = min_buffer_size
        self.mini_batch_size = mini_batch_size
        self.target_update_steps = target_update_steps
        self.iteration_count = 0

        # Create networks
        self.target_network = DQNNetwork(self.state_size, self.num_actions)
        self.network = copy.deepcopy(self.target_network)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def select_action(self, state) -> int:
        state = torch.tensor([state], dtype=torch.float32)
        actions = self.network(state)
        action = self.epsilon_greedy(actions.tolist(), self.epsilon)
        return action

    def learn(self, trajectory: tuple) -> None:
        # Add collected experiences to replay buffer
        self.replay_buffer.extend(trajectory)

        # If we have not collected enough experience then return and collect more
        if len(self.replay_buffer) < self.min_buffer_size:
            return

        # Samples a batch of data
        batch_data = self.replay_buffer.sample(self.mini_batch_size)
        batch_data = torch.tensor(batch_data, dtype=torch.float32)
        st, at, rt1, st1, done = torch.unbind(batch_data, dim=1)

        # Compute TD Target Values
        target_values = self.target_network(st1.unsqueeze(-1))
        td_target = rt1 + (1-done) * self.gamma * torch.max(target_values, dim=1)[0]

        # Compute forward pass of first network
        q = self.network(st.unsqueeze(-1))
        qsa = torch.gather(q, dim=1, index=at.to(torch.int64).unsqueeze(-1)).squeeze()

        # Compute the MSE loss
        # loss = torch.mean((td_target - qsa)**2)
        loss = torch.nn.functional.smooth_l1_loss(qsa, td_target)

        # Backprop the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()

        # if the number of steps is reach update the target weights
        if self.iteration_count == self.target_update_steps:
            self.target_network.load_state_dict(self.network.state_dict())
            self.iteration_count = 0

        self.iteration_count += 1

    def epsilon_greedy(self, actions: list[float], epsilon: float) -> int:
        r"""
        Epsilon-greedy policy chooses the action with the highest value and samples all actions randomly with probability epsilon.

        If :math:`b > \epsilon` , use Greedy; otherwise choose randomly from amount all actions

        Args:
            actions (list[int]): list of action values
            epsilon (float): probability of choosing a random exploratory action

        Returns:
            int: Selected action
        """
        # Select action with largest value, choose randomly if there is more than 1
        a_star = np.where(actions == np.max(actions))[0]
        if len(a_star) > 1:
            a_star = np.random.choice(a_star)
        else:
            a_star = a_star[0]

        if np.random.random() > epsilon:
            action = a_star
        else:
            # Choose random from actions
            action = np.random.choice(len(actions))

        return action

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def extend(self, experience):
        # self.memory.extend(experience)
        self.memory.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        return experiences

    def __len__(self):
        return len(self.memory)


