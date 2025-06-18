"""
Proximal Policy Optimization (PPO)

Reference:
[1] https://arxiv.org/abs/1707.06347
"""
import torch
import torch.nn.functional as F
from typing import Optional, List
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.policies import ActorCriticPolicy
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator


class PPO(BaseAgent):
    """
    Proximal Policy Optimization (PPO)

    """
    def __init__(self,
                 env_params: EnvParams,
                 policy: Optional[ActorCriticPolicy] = None,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 learning_rate: float = 3e-4,
                 num_optim_steps: int = 10,
                 mini_batch_size: int = 32,
                 ) -> None:
        super().__init__(policy=policy)
        self.env_params = env_params
        self.policy = policy if policy is not None else ActorCriticPolicy(env_params)
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_optim_steps = num_optim_steps
        self.mini_batch_size = mini_batch_size

        # Configure Replay Buffer
        # Configure optimizers
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    @staticmethod
    def compute_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Compute rewards-to-go for each timestep in an episode.
        """
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

    @staticmethod
    def compute_advantages(returns: torch.Tensor, values: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        advantages = returns - values
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    @staticmethod
    def compute_loss(self,
                     batch_experience: TensorDict,
                     batch_returns: torch.Tensor,
                     batch_advantages: torch.Tensor,
                     batch_action_log_probs: torch.Tensor,
                     ) -> List:
        batch_actions = batch_experience['action'].clone()

        # Recompute log probs and value estimates
        _ = self.policy.get_action(batch_experience)
        new_values = self.policy.get_value_estimates()
        new_action_log_probs = self.policy.get_log_probs(batch_actions)

        ratio = torch.exp(new_action_log_probs - batch_action_log_probs)
        clipped_adv = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
        clip_loss = -torch.min(ratio * batch_advantages, clipped_adv).mean()

        value_loss = F.mse_loss(new_values, batch_returns)

        return [clip_loss, value_loss]

    def predict(self, state):
        return super().predict(state)
    
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              logging_freq: int = 1000,
              evaluator: Evaluator = Evaluator(),
              eval_freq: int = 1000
              ) -> None:
        """
        Train the PPO agent.

        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of steps to train for.
            schedulers (Optional[List[ParameterScheduler]]): Learning rate schedulers.
            logger (Optional[Logger]): Logger for training metrics.
            logging_freq (int): Frequency of logging.
            evaluator (Optional[Any]): Evaluator for performance evaluation.
            eval_freq (int): Frequency of evaluation.
        """
        logger = logger or Logger.create('blank')
        progress_bar = ProgressBar(total_steps=total_steps)
        num_steps = 0

        # Make collector

        while num_steps < total_steps:
            # Update Schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)

            # Collect experience
            # Compute the Returns - Rewards to go
            # Compute Advantages

            # Optimization Loop
            for _ in range(self.num_optim_steps):
                for i in range(0, self.replay_buffer.get_size(), self.mini_batch_size):
                    # Get batch data
                    # Compute loss
                    loss = None

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients if necessary
                    self.optimizer.step()

            # Update progress bar
            # Log metrics







