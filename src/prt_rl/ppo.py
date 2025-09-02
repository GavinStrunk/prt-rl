"""
Proximal Policy Optimization (PPO)

Reference:
[1] https://arxiv.org/abs/1707.06347
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.buffers import RolloutBuffer
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils

from prt_rl.common.policies import ActorCriticPolicy

@dataclass
class PPOConfig:
    """
    Configuration for the PPO agent.

    Args:
        steps_per_batch (int): Number of steps to collect per batch.
        mini_batch_size (int): Size of mini-batches for optimization.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Clipping parameter for PPO.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        entropy_coef (float): Coefficient for the entropy term in the loss function.
        value_coef (float): Coefficient for the value loss term in the loss function.
        num_optim_steps (int): Number of optimization steps per batch.
        normalize_advantages (bool): Whether to normalize advantages.
    """
    steps_per_batch: int = 2048
    mini_batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.1
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_optim_steps: int = 10
    normalize_advantages: bool = False


class PPO(BaseAgent):
    """
    Proximal Policy Optimization (PPO)

    Args:
        env_params (EnvParams): Environment parameters.
        policy (ActorCriticPolicy | None): Policy to use. If None, a default ActorCriticPolicy will be created.
        steps_per_batch (int): Number of steps to collect per batch.
        mini_batch_size (int): Size of mini-batches for optimization.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Clipping parameter for PPO.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        entropy_coef (float): Coefficient for the entropy term in the loss function.
        value_coef (float): Coefficient for the value loss term in the loss function.
        num_optim_steps (int): Number of optimization steps per batch.
        normalize_advantages (bool): Whether to normalize advantages.
        device (str): Device to run the computations on ('cpu' or 'cuda').
    """
    def __init__(self,
                 env_params: EnvParams,
                 policy: ActorCriticPolicy | None = None,
                 config: PPOConfig = PPOConfig(),
                 device: str = 'cpu',
                 ) -> None:
        super().__init__()
        self.env_params = env_params
        self.config = config
        self.device = torch.device(device)

        self.policy = policy if policy is not None else ActorCriticPolicy(env_params=env_params)
        self.policy.to(self.device)

        # Configure optimizers
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

    def predict(self, 
                state: torch.Tensor, 
                deterministic: bool = False
                ) -> torch.Tensor:
        """
        Predict the action based on the current state.

        Args:
            state (torch.Tensor): Current state of the environment.
            deterministic (bool): If True, use the deterministic policy. Otherwise, sample from the policy.

        Returns:
            torch.Tensor: Predicted action.
        """
        with torch.no_grad():
            return self.policy(state, deterministic=deterministic)  
    
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              evaluator: Optional[Evaluator] = None,
              show_progress: bool = True
              ) -> None:
        """
        Train the PPO agent.

        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of steps to train for.
            schedulers (Optional[List[ParameterScheduler]]): Learning rate schedulers.
            logger (Optional[Logger]): Logger for training metrics.
            evaluator (Optional[Any]): Evaluator for performance evaluation.
            show_progress (bool): If True, show a progress bar during training.
        """
        logger = logger or Logger.create('blank')

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        num_steps = 0

        # Make collector and do not flatten the experience so the shape is (N, T, ...)
        collector = ParallelCollector(env=env, logger=logger, flatten=False)
        rollout_buffer = RolloutBuffer(capacity=self.config.steps_per_batch, device=self.device)

        while num_steps < total_steps:
            # Update Schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)

            # Collect experience dictionary with shape (N, T, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.config.steps_per_batch)

            # Compute Advantages and Returns under the current policy
            advantages, returns = utils.generalized_advantage_estimates(
                rewards=experience['reward'],
                values=experience['value_est'],
                dones=experience['done'],
                last_values=experience['last_value_est'],
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda
            )
            
            if self.config.normalize_advantages:
                advantages = utils.normalize_advantages(advantages)

            experience['advantages'] = advantages.detach()
            experience['returns'] = returns.detach()

            # Flatten the experience batch (N, T, ...) -> (N*T, ...) and remove the last_value_est key because we don't need it anymore
            experience = {k: v.reshape(-1, *v.shape[2:]) for k, v in experience.items() if k != 'last_value_est'}
            num_steps += experience['state'].shape[0]

            # Add experience to the rollout buffer
            rollout_buffer.add(experience)

            # Optimization Loop
            clip_losses = []
            entropy_losses = []
            value_losses = []
            losses = []
            for _ in range(self.config.num_optim_steps):
                for batch in rollout_buffer.get_batches(batch_size=self.config.mini_batch_size):
                    new_value_est, new_log_prob, entropy = self.policy.evaluate_actions(batch['state'], batch['action'])
                    old_log_prob = batch['log_prob'].detach()

                    # Ratio between new and old policy
                    ratio = torch.exp(new_log_prob.detach() - old_log_prob)

                    # Clipped surrogate loss
                    batch_advantages = batch['advantages']
                    clip_loss = batch_advantages * ratio
                    clip_loss2 = batch_advantages * torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
                    clip_loss = -torch.min(clip_loss, clip_loss2).mean()

                    entropy_loss = -entropy.mean()

                    value_loss = F.mse_loss(new_value_est, batch['returns'])

                    loss = clip_loss + self.config.entropy_coef*entropy_loss + self.config.value_coef * value_loss
                    
                    clip_losses.append(clip_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    value_losses.append(value_loss.item())
                    losses.append(loss.item())

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            # Clear the buffer after optimization
            rollout_buffer.clear()

            # Update progress bar
            if show_progress:
                tracker = collector.get_metric_tracker()
                progress_bar.update(current_step=num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}, "
                                                                   f"Episode Length: {tracker.last_episode_length}, "
                                                                   f"Loss: {np.mean(losses):.4f},")
            # Log metrics
            if logger.should_log(num_steps):
                logger.log_scalar('clip_loss', np.mean(clip_losses), num_steps)
                logger.log_scalar('entropy_loss', np.mean(entropy_losses), num_steps)
                logger.log_scalar('value_loss', np.mean(value_losses), num_steps)
                logger.log_scalar('loss', np.mean(losses), num_steps)
                # logger.log_scalar('episode_reward', collector.previous_episode_reward, num_steps)
                # logger.log_scalar('episode_length', collector.previous_episode_length, num_steps)

            if evaluator is not None:
                # Evaluate the agent periodically
                evaluator.evaluate(agent=self.policy, iteration=num_steps)

        if evaluator is not None:
            evaluator.close()





