"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils

import copy
import numpy as np
from dataclasses import dataclass
from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.policies import BasePolicy, ContinuousPolicy, StateActionCritic


@dataclass
class TD3Config:
    """
    Configuration for the TD3 agent.

    Args:
        buffer_size (int): Size of the replay buffer.
        min_buffer_size (int): Minimum size of the replay buffer before training starts.
        steps_per_batch (int): Number of steps to collect per batch.
        mini_batch_size (int): Size of the mini-batch sample for each gradient update.
        gradient_steps (int): Number of gradient steps to take per training iteration.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        exploration_noise (float): Standard deviation of Gaussian noise added to actions for exploration.
        policy_noise (float): Standard deviation of noise added to the target policy's actions.
        noise_clip (float): Maximum absolute value of noise added to the target policy's actions.
        delay_freq (int): Frequency of delayed policy updates.
        tau (float): Polyak averaging factor for target networks.
        num_critics (int): Number of critic networks to use.
    """
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    steps_per_batch: int = 1
    mini_batch_size: int = 256
    gradient_steps: int = 1
    learning_rate: float = 1e-3
    gamma: float = 0.99
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    delay_freq: int = 2
    tau: float = 0.005
    num_critics: int = 2

class TD3Policy(BasePolicy):
    """
    TD3 Policy

    This class implements the TD3 policy, which consists of an actor network and multiple critic networks.
    The actor network is used to select actions, while the critic networks are used to evaluate the actions.
    The policy can share the encoder with the actor and critic networks if specified.

    Args:
        env_params (EnvParams): Environment parameters.
        num_critics (int): Number of critic networks to use. Default is 2.
        actor (Optional[ContinuousPolicy]): Custom actor network. If None, a default actor will be created.
        critic (Optional[StateActionCritic]): Custom critic network. If None, a default critic will be created.
        share_encoder (bool): Whether to share the encoder between actor and critic networks. Default is True.
        device (str): Device to run the policy on ('cpu' or 'cuda'). Default is 'cpu'.
    """
    def __init__(self, 
                 env_params: EnvParams, 
                 num_critics: int = 2,
                 actor: Optional[ContinuousPolicy] = None,
                 critic: Optional[StateActionCritic] = None,
                 share_encoder: bool = True,
                 device: str='cpu'
                 ) -> None:
        super().__init__(env_params=env_params)
        self.num_critics = num_critics
        self.share_encoder = share_encoder
        self.device = torch.device(device)

        # Create actor and target actor networks
        self.actor = actor if actor is not None else ContinuousPolicy(env_params=env_params)
        self.actor.to(self.device)
        actor_encoder = self.actor.get_encoder()
        if not self.share_encoder and actor_encoder is not None:
            actor_encoder = copy.deepcopy(actor_encoder)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.to(self.device)

        self.critic = critic if critic is not None else StateActionCritic(env_params=env_params, num_critics=num_critics, encoder=actor_encoder)
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
        action = self.actor(state, deterministic=deterministic)

        value_estimates = self.critic(state, action)
        value_estimates = torch.stack(value_estimates, dim=1)  # Shape (B, C, 1) where C is the number of critics
        return action, value_estimates, None
    
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
    

class TD3(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)

    This class implements the TD3 algorithm, which is an off-policy actor-critic algorithm for continuous action spaces.

    Args:
        env_params (EnvParams): Environment parameters.
        policy (TD3Policy | None): Custom TD3 policy. If None, a default TD3 policy will be created.
        config (TD3Config): Configuration for the TD3 agent.
        device (str): Device to run the agent on ('cpu' or 'cuda'). Default is 'cpu'.
    """
    def __init__(self,
                 env_params: EnvParams,
                 policy: TD3Policy | None = None,
                 config: TD3Config = TD3Config(),
                 device: str = 'cpu',
                 ) -> None:
        super().__init__()
        self.env_params = env_params
        self.config = config
        self.device = torch.device(device)

        self.policy = policy if policy is not None else TD3Policy(env_params=env_params, num_critics=self.config.num_critics, device=device)
        self.policy.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.config.learning_rate) for critic in self.policy.critic.critics
        ]

        self.action_min = torch.tensor(env_params.action_min, device=self.device, dtype=torch.float32)
        self.action_max = torch.tensor(env_params.action_max, device=self.device, dtype=torch.float32)

    def predict(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Perform an action based on the current state.

        Args:
            state (torch.Tensor): The current state of the environment.
            deterministic (bool): If True, the agent will select actions deterministically.

        Returns:
            torch.Tensor: The action to be taken.
        """
        with torch.no_grad():
            action = self.policy(state)
            if not deterministic:
                # Add noise to the action for exploration
                noise = utils.gaussian_noise(mean=0, std=self.config.exploration_noise, shape=action.shape, device=self.device)
                action = action + noise
                action = action.clamp(self.action_min, self.action_max)
        return action
        
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              evaluator: Optional[Evaluator] = None,
              show_progress: bool = True
              ) -> None:
        """
        Update the agent's knowledge based on the action taken and the received reward.
        This method should implement the TD3 training loop.

        Args:
            env: The environment to interact with.
            total_steps: Total number of steps to train the agent.
            schedulers: Optional list of parameter schedulers.
            logger: Optional logger for logging training progress.
            evaluator: Evaluator for evaluating the agent's performance.
            show_progress: If True, show a progress bar during training.
        """
        logger = logger or Logger.create('blank')

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        num_steps = 0
        num_gradient_steps = 0

        # Make collector and flatten the experience so the shape is (B, ...)
        collector = ParallelCollector(env=env, logger=logger, flatten=True)
        replay_buffer = ReplayBuffer(capacity=self.config.buffer_size, device=self.device)

        while num_steps < total_steps:
            # Update Schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)

            # Collect experience dictionary with shape (B, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.config.steps_per_batch)
            num_steps += experience['state'].shape[0]

            # Store experience in replay buffer
            replay_buffer.add(experience)

            # Collect a minimum number of steps in the replay buffer before training
            if replay_buffer.get_size() < self.config.min_buffer_size:
                if show_progress:
                    progress_bar.update(current_step=num_steps, desc="Collecting initial experience...")
                continue

            actor_losses = []
            critics_losses = []
            for _ in range(self.config.gradient_steps):
                num_gradient_steps += 1

                # Sample a batch of experiences from the replay buffer
                batch = replay_buffer.sample(batch_size=self.config.mini_batch_size)

                # Compute current policy's action and target
                # We compute the target y values without gradients because they will be used to compute the loss for each critic
                # so an error will be raised for trying to backpropagate through y more than once.
                with torch.no_grad():
                    # Compute the policies next action with noise and clip to ensure it does not exceed action bounds
                    noise = utils.gaussian_noise(mean=0, std=self.config.policy_noise, shape=batch['action'].shape, device=self.device)
                    noise_clipped = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
                    next_action = (self.policy.actor_target(batch['next_state']) + noise_clipped) #.clamp(self.env_params.action_min, self.env_params.action_max)

                    # Compute the Q-Values for all the critics - shape (B, C, 1) -> (B, C)
                    next_q_values = self.policy.get_target_q_values(batch['next_state'], next_action).squeeze(-1) 

                    # Use the minimum Q-Value across critics for the target
                    next_q_values = torch.min(next_q_values, dim=1, keepdim=True)[0] 

                    # Compute the target Q-Value
                    y = batch['reward'] + self.config.gamma * (1 - batch['done'].float()) * next_q_values

                # Update critics
                for i in range(self.policy.num_critics):
                    # Compute critics loss
                    q_i = self.policy.get_q_values(batch['state'].detach(), batch['action'].detach(), index=i)
                    critic_loss = F.mse_loss(y, q_i)
                    critics_losses.append(critic_loss.item())

                    self.critic_optimizers[i].zero_grad()
                    critic_loss.backward()
                    self.critic_optimizers[i].step()

                # Delayed policy update 
                if num_gradient_steps % self.config.delay_freq == 0:
                    # Compute actor loss
                    actor_loss = -self.policy.get_q_values(state=batch['state'], action=self.policy.actor(batch['state']), index=0).mean()
                    actor_losses.append(actor_loss.item())

                    # Take a gradient step on the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Update target networks
                    utils.polyak_update(self.policy.actor_target, self.policy.actor, tau=self.config.tau)
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