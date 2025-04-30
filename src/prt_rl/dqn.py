import copy
import torch
from typing import Optional, List
from tensordict import TensorDict
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.trainers import ANNTrainer
from prt_rl.common.replay_buffers import ReplayBuffer
from prt_rl.common.decision_functions import DecisionFunction
from prt_rl.common.policy import QNetworkPolicy
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.metrics import MetricTracker

class DQN(ANNTrainer):
    """
    Deep Q Network Algorithm

    Algorithm Steps:
        - Initialize replay memory D to capacity N
        - Initialize action-value function Q with random weights
        - for episode = 1, M do
            - Initialize sequence s = {x} and preprocessed sequenced phi = phi(s)
            - for t = 1,T do
                - Select action with epsilon greedy decision function
                - Collect experience from environment
                - Store experience in replay buffer
                - Sample random minibatch of transitions from D
                - Set y for terminal or non-terminal states
                - Perform gradient descent step

    References:
        [1] https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 num_envs: int = 1,
                 decision_function: Optional[DecisionFunction] = None,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 schedulers: Optional[List[ParameterScheduler]] = None,
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
        self.alpha = alpha
        self.gamma = gamma
        self.min_buffer_size = min_buffer_size
        self.mini_batch_size = mini_batch_size
        self.target_update_steps = target_update_steps
        self.device = device
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        policy = QNetworkPolicy(
            env_params=self.env_params,
            num_envs=1,
            decision_function=decision_function,
            device=device
        )
        super(DQN, self).__init__(env=env, policy=policy, logger=logger, schedulers=schedulers, metric_tracker=metric_tracker)

        self.target_network = copy.deepcopy(self.get_policy_network())
        self.optimizer = torch.optim.Adam(self.policy.q_network.parameters(), lr=self.alpha)
        self.iteration_counter = 0

    def update_policy(self, experience: TensorDict) -> None:
        # Add experience to replay buffer
        self.replay_buffer.add(experience)

        # Collect more experience if there is not a minimum number
        if len(self.replay_buffer) < self.min_buffer_size:
            return

        # Sample a batch of data
        batch_data = self.replay_buffer.sample(self.mini_batch_size)
        st = batch_data['observation']
        at = batch_data['action']
        st1 = batch_data['next', 'observation']
        rt1 = batch_data['next', 'reward']
        done = batch_data['next', 'done']

        # Compute TD Target values
        target_values = self.target_network(st1)
        td_target = rt1 + (1 - done) * self.gamma * torch.max(target_values, dim=1)[0]

        # Compute Policy values
        q = self.policy.q_network(st)
        qsa = torch.gather(q, dim=1, index=at)

        # Compute Loss
        loss = torch.mean((td_target - qsa)**2)

        # Optimize the policy model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.policy.q_network.parameters(), 1.0)

        # Update target network parameters
        # @todo implement option for polyak update
        if self.iteration_counter == self.target_update_steps:
            self.target_network.load_state_dict(self.policy.q_network.state_dict())
            self.iteration_counter = 0
        self.iteration_counter += 1
