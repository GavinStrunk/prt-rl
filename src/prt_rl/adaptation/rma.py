"""
Rapid Motor Adaptation
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from prt_rl.agent import BaseAgent
from prt_rl.common.policies import BasePolicy, DistributionPolicy, ValueCritic
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator

import numpy as np
import torch.nn.functional as F
from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.buffers import RolloutBuffer
import prt_rl.common.utils as utils
from prt_rl import PPO


@dataclass
class RMAConfig:
    steps_per_batch: int = 80_000
    mini_batch_size: int = 4
    learning_rate: float = 5e-4
    gamma: float = 0.99
    epsilon: float = 0.1
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_optim_steps: int = 10
    normalize_advantages: bool = False
    adapt_learning_rate: float = 5e-4
    adapt_batch_size: int = 80_000
    adapt_mini_batch_size: int = 4


class RMAPolicy(BasePolicy):
    """
    Rapid Motor Adaptation (RMA) Policy.
    
    The policy takes the current state, previous action, and environment encoding as input.
    """
    def __init__(self, 
                 env_params: EnvParams,
                 extrinsics_dim: int = 8,
                 time_steps: int = 50
                 ) -> None:
        super().__init__(env_params)
        self.extrinsics_dim = extrinsics_dim
        self.time_steps = time_steps

        # Update the 
        policy_dim = env_params.observation_shape[0] + env_params.action_shape[0] + extrinsics_dim
        env_params.observation_shape = (policy_dim, )

        self.actor = DistributionPolicy(
            env_params=env_params,
            policy_kwargs={'network_arch': [128, 128]}
        )

        self.critic = ValueCritic(
            env_params=env_params,
            critic_head_kwargs={'network_arch': [128, 128]},
            )
        
        self.adaptation_module = AdaptationModule(
            env_dim=(self.env_params.observation_shape[0] + self.env_params.action_shape[0]) * self.time_steps,
            extrinsics_dim=self.extrinsics_dim
        )       

        env_dim = (env_params.observation_shape[0] + env_params.action_shape[0]) * time_steps

    def forward(self, 
                state: torch.Tensor, 
                deterministic: bool = False
                ) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            state (torch.Tensor): The current state of the environment.
            deterministic (bool): If True, the action will be selected deterministically.

        Returns:
            The action to be taken.
        """
        pass

    def predict(self, 
                state: torch.Tensor, 
                deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the action based on the current state.

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the chosen action, value estimate, and action log probability.
                - action (torch.Tensor): Tensor with the chosen action. Shape (B, action_dim)
                - value_estimate (torch.Tensor): Tensor with the estimated value of the state. Shape (B, C, 1) where C is the number of critics
                - log_prob (torch.Tensor): None
        """

class EnvironmentEncoder(nn.Module):
    def __init__(self, 
                 env_dim: int,
                 extrinsics_dim: int
                 ) -> None:
        super().__init__()  

        self.layers = nn.Sequential(
            nn.Linear(env_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, extrinsics_dim),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class AdaptationModule(nn.Module):
    def __init__(self, 
                 env_dim: int,
                 extrinsics_dim: int
                 ) -> None:
        super().__init__()  

        self.layers = nn.Sequential(
            nn.Linear(env_dim, 32),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(32*3, extrinsics_dim),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class RMA(BaseAgent):
    def __init__(self,
                 agent: PPO,
                 config: RMAConfig,
                 device: str = 'cpu'
                 ) -> None:
        super().__init__()
        self.config = config
        self.agent = agent
        self.device = device

    def predict(self, 
                state: torch.Tensor, 
                deterministic: bool = False
                ) -> torch.Tensor:
        """
        Predict the action based on the current state.
        
        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.
        
        Returns:
            torch.Tensor: Tensor with the chosen action. Shape (B, action_dim)
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
        Train the SAC agent.
        
        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of environment steps to train for.
            schedulers (List[ParameterScheduler] | None): List of parameter schedulers to update during training.
            logger (Logger | None): Logger for logging training metrics. If None, a default logger will be created.
            evaluator (Evaluator | None): Evaluator for periodic evaluation during training.
            show_progress (bool): If True, display a progress bar during training.
        """

        # Construct the environment (extrinsics) encoder
        env_encoder = EnvironmentEncoder(
            env_dim=(self.policy.env_params.observation_shape[0] + self.policy.env_params.action_shape[0]) * self.policy.time_steps,
            extrinsics_dim=self.policy.extrinsics_dim
        ).to(self.device)

        actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.config.learning_rate)
        critic_optimizer = torch.optim.Adam([self.policy.critic.parameters(), env_encoder.parameters()], lr=self.config.learning_rate)

        # Use PPO Training with Extrinsics Encoder + Policy
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

                    # Optimize the critic + env encoder
                    extrinsic_features = env_encoder(batch['state'])
                    new_value_est = self.policy.critic(extrinsic_features)

                    value_loss = F.mse_loss(new_value_est, batch['returns'])
                    critic_optimizer.zero_grad()
                    value_loss.backward()
                    critic_optimizer.step()

                    value_losses.append(value_loss.item())

                    # Optimize the actor
                    with torch.no_grad():
                        extrinsic_features = env_encoder(batch['state'])
                    
                    new_log_probs, entropy = self.policy.actor.evaluate_actions(extrinsic_features, batch['action'])
                    # new_value_est, new_log_prob, entropy = self.policy.evaluate_actions(batch['state'], batch['action'])
                    
                    # We detach the old log probs so we only computed gradients for the current policies parameters
                    old_log_prob = batch['log_prob'].detach()

                    # Ratio between new and old policy
                    ratio = torch.exp(new_log_probs - old_log_prob)

                    # Clipped surrogate loss
                    batch_advantages = batch['advantages']
                    clip_loss = batch_advantages * ratio
                    clip_loss2 = batch_advantages * torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
                    clip_loss = -torch.min(clip_loss, clip_loss2).mean()

                    entropy_loss = -entropy.mean()

                    loss = clip_loss + self.config.entropy_coef*entropy_loss
                    
                    clip_losses.append(clip_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    losses.append(loss.item())

                    # Optimize
                    actor_optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    actor_optimizer.step()

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

    def train_adaptation(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: List[ParameterScheduler] = [],              
              logger: Optional[Logger] = None,
                         ) -> None:
        """
        Train the adaptation module for rapid motor adaptation.

        DAgger style supervised learning training.
        """

        adaptation_optimizer = torch.optim.Adam(self.policy.adaptation_module.parameters(), lr=self.config.adapt_learning_rate)

        # Create collector and replay buffer

        num_steps = 0
        while num_steps < total_steps:
            # Update schedulers if any
            for scheduler in schedulers:
                scheduler.update(current_step=num_steps)

            # Collect experience using the trained base policy and environment encoder
            # Get Data [S_t, A_t, z_t]_i
            # Update number of steps

            # Stack states and actions over time window
            target_extrinsics = None

            # For batches
            # predict extrinsics from adaptation module
            batch_data = None
            predicted_extrinsics = self.adaptation_module(batch_data)

            adapt_loss = F.mse_loss(predicted_extrinsics, target_extrinsics)

            # Run supervised optimization
            self.adaptation_optimizer.zero_grad()
            adapt_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adaptation_module.parameters(), 1.0)
            self.adaptation_optimizer.step()
            
    def fine_tune():
        """
        Fine-tune the policy on a new task or environment.

        This is the third phase of RMA where the base policy is fine-tuned with the trained adaptation module.
        """
        pass