"""
Soft Actor-Critic (SAC)
"""
from dataclasses import dataclass
import torch
from typing import Optional, List, Tuple
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator

import copy
import numpy as np
from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.policies import DistributionPolicy, StateActionCritic, BasePolicy
import prt_rl.common.utils as utils

@dataclass
class SACConfig:
    buffer_size: int = 1000000
    min_buffer_size: int = 100
    steps_per_batch: int = 256
    mini_batch_size: int = 256
    gradient_steps: int = 1
    learning_rate: float = 3e-4
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    
class SACPolicy(BasePolicy):
    """
    Soft Actor-Critic (SAC) policy class.
    
    Args:
        env_params (EnvParams): Environment parameters.
        num_critics (int): Number of critics to use in the SAC algorithm.
        actor (DistributionPolicy | None): Actor policy. If None, a default DistributionPolicy will be created.
        critic (StateActionCritic | None): Critic network. If None, a default StateActionCritic will be created.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
    """
    def __init__(self,
                env_params: EnvParams,
                num_critics: int = 2,
                actor: DistributionPolicy | None = None,
                critic: StateActionCritic | None = None,
                device: str = 'cpu'
                ) -> None:
        super().__init__(env_params=env_params)
        self.num_critics = num_critics
        self.device = torch.device(device)

        self.actor = actor if actor is not None else DistributionPolicy(env_params=env_params)
        self.actor.to(self.device)

        self.critic = critic if critic is not None else StateActionCritic(env_params=env_params, num_critics=num_critics)
        self.critic.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.to(self.device)

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
        return self.actor(state, deterministic=deterministic)

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
        action, _, log_probs = self.actor.predict(state, deterministic=deterministic)     

        value_estimates = self.critic(state, action)
        value_estimates = torch.stack(value_estimates, dim=1)  # Shape (B, C, 1) where C is the number of critics
        return action, value_estimates, log_probs
    
    def get_q_values(self,
                     state: torch.Tensor,
                     action: torch.Tensor,
                     index: Optional[int] = None
                     ) -> torch.Tensor:
        """
        Get Q-values from all critics for the given state-action pairs.

        Args:
            state (torch.Tensor): Current state tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Tensor containing Q-values from all critics. Shape (B, C, 1) where C is the number of critics.
        """
        if index is None:
            q_values = self.critic(state, action)
            q_values = torch.stack(q_values, dim=1)  # Shape (B, C, 1) where C is the number of critics
        else:
            q_values = self.critic.forward_indexed(index, state, action)
        return q_values
    
    def get_target_q_values(self,
                            state: torch.Tensor,
                            action: torch.Tensor,
                            ) -> torch.Tensor:
        """
        Get target Q-values from all target critics for the given state-action pairs.

        Args:
            state (torch.Tensor): Current state tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Tensor containing target Q-values from all critics. Shape (B, C, 1) where C is the number of critics.
        """
        q_values = self.critic_target(state, action)
        q_values = torch.stack(q_values, dim=1)  # Shape (B, C, 1) where C is the number of critics
        return q_values


class SAC(BaseAgent):
    """
    Soft Actor-Critic (SAC) agent.

    Args:
        env_params (EnvParams): Environment parameters.
        policy (SACPolicy | None): Policy to use. If None, a default SACPolicy will be created.
        config (SACConfig): Configuration for the SAC agent.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').

    References:
        [1] https://arxiv.org/pdf/1812.05905    
    """
    def __init__(self, 
                 env_params: EnvParams, 
                 policy: SACPolicy | None = None,
                 config: SACConfig = SACConfig(), 
                 device: str = 'cpu'
                 ) -> None:
        super().__init__()
        self.env_params = env_params
        self.config = config
        self.device = torch.device(device)
        
        self.policy = policy if policy is not None else SACPolicy(env_params=env_params, device=device) 
        self.policy.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.config.learning_rate) for critic in self.policy.critic.critics
        ]
        # Initialize the entropy coefficient and target
        self.target_entropy = -float(env_params.action_len)
        self.entropy_coeff = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.entropy_optimizer = torch.optim.Adam([self.entropy_coeff], lr=1)


    def predict(self, 
                state: torch.Tensor, 
                deterministic: bool = False
                ) -> torch.Tensor:
        
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
        logger = logger or Logger.create('blank')

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        num_steps = 0

        collector = ParallelCollector(env=env, logger=logger, flatten=True)   
        replay_buffer = ReplayBuffer(capacity=self.config.buffer_size, device=self.device)

        while num_steps < total_steps:
            # Update schedulers
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)  

            # Collect experience dictionary with shape (B, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.config.steps_per_batch, bootstrap=False)
            num_steps += experience['state'].shape[0]
            replay_buffer.add(experience)

            actor_losses = []
            critics_losses = []
            entropy_losses = []
            entropy_coeffs = []
            for _ in range(self.config.gradient_steps):
                # Sample a mini-batch from the replay buffer
                mini_batch = replay_buffer.sample(batch_size=self.config.mini_batch_size)

                # Compute the current policy's action and log probability
                current_action, _, current_log_prob = self.policy.predict(mini_batch['state'])

                # Entropy coefficient optimization
                entropy_loss = -(self.entropy_coeff * (current_log_prob + self.target_entropy).detach()).mean()
                entropy_losses.append(entropy_loss.item())
                self.entropy_optimizer.zero_grad()
                entropy_loss.backward()
                self.entropy_optimizer.step()
                entropy_coeffs.append(self.entropy_coeff.item())

                # Compute the target values from the current policy
                with torch.no_grad():
                    # Select next action based on current policy
                    next_action, _, next_log_prob = self.policy.predict(mini_batch['next_state'])

                    # Compute the Q-values for all critics
                    next_q_values = self.policy.get_target_q_values(state=mini_batch['next_state'], action=next_action).squeeze(-1)
                    next_q_values = torch.min(next_q_values, dim=1, keepdim=True)[0]

                    # Add the entropy term to the target Q-values
                    next_q_values += -self.entropy_coeff * next_log_prob.reshape(-1, 1)

                    # Compute the discounted target Q-values
                    y = mini_batch['reward'] + (1 - mini_batch['done'].float()) * self.config.gamma * next_q_values

                # Update critics
                for i in range(self.policy.num_critics):
                    q_i = self.policy.get_q_values(state=mini_batch['state'].detach(), action=mini_batch['action'].detach(), index=i)
                    critic_loss = torch.nn.functional.mse_loss(q_i, y)
                    critics_losses.append(critic_loss.item())

                    self.critic_optimizers[i].zero_grad()
                    critic_loss.backward()
                    self.critic_optimizers[i].step()

                # Compute Actor loss
                q_values_pi = self.policy.get_q_values(state=mini_batch['state'], action=current_action)
                q_values_pi = torch.min(q_values_pi, dim=1, keepdim=True)[0]
                actor_loss = (self.entropy_coeff * current_log_prob - q_values_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update target networks
                for i in range(self.policy.num_critics):
                    utils.polyak_update(self.policy.critic_target.critics[i], self.policy.critic.critics[i], tau=self.config.tau)   

            if show_progress:
                tracker = collector.get_metric_tracker()
                progress_bar.update(current_step=num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}"
                                                                f" Episode Length: {tracker.last_episode_length}"
                                                                f" Episode number: {tracker.episode_count}"
                                                                f" Actor Loss: {np.mean(actor_losses):.4f}"
                                                                )

            if logger.should_log(num_steps):
                logger.log_scalar('actor_loss', np.mean(actor_losses), num_steps)
                for i in range(self.policy.num_critics):
                    logger.log_scalar(f'critic{i}_loss', np.mean(critics_losses[i]), num_steps)

            if evaluator is not None:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)

        if evaluator is not None:
            evaluator.close()                                 
