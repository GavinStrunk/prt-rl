"""
Proximal Policy Optimization (PPO)

Reference:
[1] https://arxiv.org/abs/1707.06347
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.policies import ActorCriticPolicy
from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.buffers import RolloutBuffer
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
                 gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 num_optim_steps: int = 10,
                 mini_batch_size: int = 32,
                 steps_per_batch: int = 2048,
                 device: str = 'cpu',
                 ) -> None:
        super().__init__(policy=policy)
        self.env_params = env_params
        self.policy = policy if policy is not None else ActorCriticPolicy(env_params, device=device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.num_optim_steps = num_optim_steps
        self.mini_batch_size = mini_batch_size
        self.steps_per_batch = steps_per_batch
        self.device = torch.device(device)

        self.policy.to(self.device)
        # Configure optimizers
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
    
    @staticmethod
    def compute_gae_and_returns(
        rewards: torch.Tensor,  
        values: torch.Tensor,            
        dones: torch.Tensor,             
        last_values: torch.Tensor,       
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE and TD(lambda) returns for a batched rollout.

        Args:
            rewards (T, N, 1): Rewards from rollout
            values (T, N, 1): Estimated state values
            dones (T, N, 1): Done flags (1 if episode ended at step t, else 0)
            last_values (N, 1): Value estimates for final state (bootstrap)
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda

        Returns:
            advantages (T, N): Estimated advantage values
            returns (T, N): TD(lambda) returns
        """
        T, N, _ = rewards.shape

        # Concatenate last_values to get V(s_{t+1}) for final time step
        values = torch.cat([values, last_values.unsqueeze(0)], dim=0)  # shape: (T+1, N, 1)

        advantages = torch.zeros((T, N, 1), dtype=values.dtype, device=values.device)
        last_gae_lam = torch.zeros((N, 1), dtype=values.dtype, device=values.device)

        for t in reversed(range(T)):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
            last_gae_lam = delta + gamma * gae_lambda * not_done * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values[:-1]  # TD(lambda) returns
        return advantages, returns

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
        collector = ParallelCollector(env=env, logger=logger, logging_freq=logging_freq)
        rollout_buffer = RolloutBuffer(capacity=self.steps_per_batch, device=self.device)

        while num_steps < total_steps:
            # Update Schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)

            # Collect experience
            experience = collector.collect_experience(policy=self.policy, num_steps=self.steps_per_batch)
            num_steps += experience['state'].shape[0]

            # Compute Advantages
            advantages, returns = self.compute_gae_and_returns(
                rewards=experience['reward'],
                values=experience['value_est'],
                dones=experience['done'],
                last_values=experience['last_value_est'],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )
            experience['advantages'] = advantages
            experience['returns'] = returns

            # Add experience to the rollout buffer
            rollout_buffer.add(experience)

            # Optimization Loop
            clip_losses = []
            entropy_losses = []
            value_losses = []
            losses = []
            for _ in range(self.num_optim_steps):
                for batch in rollout_buffer.get_batches(batch_size=self.mini_batch_size):
                    new_value_est, new_log_prob, entropy = self.policy.evaluate_actions(batch['state'], batch['action'])

                    # Ratio between new and old policy
                    ratio = torch.exp(new_log_prob - batch['log_prob'])

                    # Clipped surrogate loss
                    batch_advantages = batch['advantages']
                    clip_loss = batch_advantages * ratio
                    clip_loss2 = batch_advantages * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    clip_loss = -torch.min(clip_loss, clip_loss2).mean()

                    entropy_loss = -entropy.mean()

                    value_loss = F.mse_loss(new_value_est, batch['returns'])

                    loss = clip_loss + self.entropy_coef*entropy_loss + self.value_coef * value_loss
                    
                    clip_losses.append(clip_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    value_losses.append(value_loss.item())
                    losses.append(loss.item())

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            # Clear the buffer after optimization
            rollout_buffer.clear()

            # Update progress bar
            progress_bar.update(current_step=num_steps, desc=f"Episode Reward: {collector.previous_episode_reward:.2f}, "
                                                                   f"Episode Length: {collector.previous_episode_length}, "
                                                                   f"Loss: {np.mean(losses):.4f},")
            # Log metrics
            logger.log_scalar('clip_loss', np.mean(clip_losses), num_steps)
            logger.log_scalar('entropy_loss', np.mean(entropy_losses), num_steps)
            logger.log_scalar('value_loss', np.mean(value_losses), num_steps)
            logger.log_scalar('loss', np.mean(losses), num_steps)
            logger.log_scalar('episode_reward', collector.previous_episode_reward, num_steps)
            logger.log_scalar('episode_length', collector.previous_episode_length, num_steps)






