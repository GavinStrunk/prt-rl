"""
Implementation of the Advantage Actor-Critic (A2C) algorithm.
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.buffers import RolloutBuffer
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils

import copy
from prt_rl.common.networks import BaseEncoder
from prt_rl.common.policies.base import BasePolicy
from prt_rl.common.policies.distribution import DistributionPolicy
from prt_rl.common.policies.value_critic import ValueCritic

@dataclass
class A2CConfig:
    """
    Configuration parameters for the A2C agent.
    
    Attributes:
        steps_per_batch (int): Number of steps to collect per training batch.
        mini_batch_size (int): Size of each mini-batch for optimization.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Clipping parameter for PPO (not used in A2C).
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        entropy_coef (float): Coefficient for the entropy bonus.
        value_coef (float): Coefficient for the value loss.
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
    normalize_advantages: bool = False

class A2CPolicy(BasePolicy):
    def __init__(self,
                 env_params: EnvParams,
                 encoder: BaseEncoder | None = None,
                 actor: DistributionPolicy | None = None,
                 critic: ValueCritic | None = None,
                 share_encoder: bool = False,
                 ) -> None:
        super().__init__(env_params=env_params)
        self.env_params = env_params
        self.encoder = encoder
        self.critic_encoder = None
        self.share_encoder = share_encoder
        
        # If no actor is provided, create a default DistributionPolicy without an encoder
        if actor is None:
            self.actor = DistributionPolicy(
                env_params=env_params,
            )
        else:
            self.actor = actor

        # If no critic is provided, create a default ValueCritic without an encoder
        if critic is None:
            self.critic = ValueCritic(
                env_params=env_params,
            )
        else:
            self.critic = critic

        # If the encoder is not shared, but one exists then make a copy for the critic
        if not share_encoder and self.encoder is not None:
            self.critic_encoder = copy.deepcopy(self.encoder)  
    
    def predict(self,
                   state: torch.Tensor,
                   deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chooses an action based on the current state and computes the value of the state.

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the chosen action, value_estimate, and aciton log probability.
        """
        if self.encoder is not None:
            latent_state = self.encoder(state)
        else:
            latent_state = state

        action, _, log_probs = self.actor.predict(latent_state, deterministic=deterministic)

        if self.critic_encoder is not None:
            critic_latent_state = self.critic_encoder(state)
        else:
            critic_latent_state = latent_state

        value = self.critic(critic_latent_state) 
        return action, value, log_probs
    
    def evaluate_actions(self,
                         state: torch.Tensor,
                         action: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the value, log probability and entropy of the given action under the policy.
        Args:
            state (torch.Tensor): Current state tensor.
            action (torch.Tensor): Action tensor to evaluate.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the value estimate, log probability, and entropy. All tensors have shape (B, 1).
        """
        if self.encoder is not None:
            latent_state = self.encoder(state)
        else:
            latent_state = state
        
        log_probs, entropy = self.actor.evaluate_actions(latent_state, action)

        if self.critic_encoder is not None:
            critic_latent_state = self.critic_encoder(state)
        else:
            critic_latent_state = latent_state
        
        value = self.critic(critic_latent_state)

        return value, log_probs, entropy

class A2C(BaseAgent):
    """
    Advantage Actor-Critic (A2C) agent implementation.
    
    Args:
        policy (A2CPolicy): The policy network used by the agent.
        config (A2CConfig): Configuration parameters for the A2C agent.
        device (str): The device to run the agent on ('cpu' or 'cuda').
    
    """
    def __init__(self,
                 policy: A2CPolicy,
                 config: A2CConfig = A2CConfig(),
                 device: str = 'cpu',
                 ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)

        self.policy = policy
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
            policy_losses = []
            entropy_losses = []
            value_losses = []
            losses = []
            for batch in rollout_buffer.get_batches(batch_size=self.config.mini_batch_size):
                new_value_est, new_log_prob, entropy = self.policy.evaluate_actions(batch['state'], batch['action'])
                
                batch_advantages = batch['advantages']
                policy_loss = -(new_log_prob * batch_advantages).mean()

                entropy_loss = -entropy.mean()
                value_loss = F.mse_loss(new_value_est, batch['returns'])
                loss = policy_loss + self.config.entropy_coef*entropy_loss + self.config.value_coef * value_loss
                
                policy_losses.append(policy_loss.item())
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
                logger.log_scalar('clip_loss', np.mean(policy_losses), num_steps)
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