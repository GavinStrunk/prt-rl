from calendar import c
import torch
import torch.nn.functional as F
from typing import Optional, List
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils

import copy
import numpy as np
from prt_rl.common.collectors import SequentialCollector
from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.policies import BasePolicy
from prt_rl.common.networks import MLP, BaseEncoder
from typing import Union, Dict, Type


class Actor(torch.nn.Module):
    def __init__(self,
                 env_params: EnvParams,
                 actor_head: Union[Type[torch.nn.Module], Dict[str, Type[torch.nn.Module]]] = MLP,
                 actor_head_kwargs: Optional[dict] = {"network_arch": [400, 300]},
                 ) -> None:
        super().__init__()
        self.env_params = env_params

        # Construct the policy head network
        self.policy_head = actor_head(
            input_dim=self.env_params.observation_shape[0],
            output_dim=self.env_params.action_len,
            **actor_head_kwargs
        )


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network.

        Args:
            state: The current state of the environment.

        Returns:
            The action to be taken.
        """
        action = self.policy_head(state)
        return action


class StateActionCritic(torch.nn.Module):
    def __init__(self, 
                 env_params: EnvParams, 
                 num_critics: int = 2,
                 critic_head: Type[torch.nn.Module] = MLP,
                 critic_head_kwargs: Optional[dict] = {"network_arch": [400, 300]},
                 ) -> None:
        super(StateActionCritic, self).__init__()
        self.env_params = env_params
        self.num_critics = num_critics

        # Initialize critics here
        self.critics = []
        for _ in range(num_critics):
            critic = critic_head(
                input_dim=self.env_params.observation_shape[0] + self.env_params.action_len,
                output_dim=1,
                **critic_head_kwargs
            )
            self.critics.append(critic)

        # Convert list to ModuleList for proper parameter management
        self.critics = torch.nn.ModuleList(self.critics)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.

        Args:
            state: The current state of the environment.
            action: The action taken in the current state.

        Returns:
            The Q-value for the given state-action pair.
        """
        # Stack the state and action tensors
        q_input = torch.cat([state, action], dim=1)

        # Return a tuple of Q-values from each critic
        return tuple(critic(q_input) for critic in self.critics)

    def forward_indexed(self, index: int, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the first critic network.

        Args:
            index (int): The index of the critic to use.
            state (torch.Tensor): The current state of the environment.
            action (torch.Tensor): The action taken in the current state.

        Returns:
            The Q-value for the given state-action pair from the first critic.
        """
        if index > self.num_critics:
            raise ValueError(f"Index {index} exceeds the number of critics {self.num_critics}.")
        
        q_input = torch.cat([state, action], dim=1)
        return self.critics[index](q_input)


class TD3Policy(BasePolicy):
    """
    Placeholder for TD3 policy class.
    This should be implemented with the actual policy logic.

    """
    def __init__(self, 
                 env_params: EnvParams, 
                 num_critics: int = 2,
                 actor: Optional[Actor] = None,
                 critic: Optional[StateActionCritic] = None,
                 device: str='cpu'
                 ) -> None:
        super().__init__(env_params=env_params)
        self.num_critics = num_critics
        self.device = torch.device(device)

        # Create actor and target actor networks
        self.actor = actor if actor is not None else Actor(env_params=env_params)
        self.actor.to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_actor.to(self.device)

        self.critic = critic if critic is not None else StateActionCritic(env_params=env_params, num_critics=num_critics)
        self.critic.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            state: The current state of the environment.

        Returns:
            The action to be taken.
        """
        return self.actor(state)

class TD3(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)

    """
    def __init__(self,
                 env_params: EnvParams,
                 policy: Optional[TD3Policy] = None,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 min_buffer_size: int = 1000,
                 gradient_steps: int = 1,
                 mini_batch_size: int = 256,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 delay_freq: int = 2,
                 tau: float = 0.005,
                 device: str = 'cpu',
                 ) -> None:
        super().__init__()
        self.env_params = env_params
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.gradient_steps = gradient_steps
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.delay_freq = delay_freq
        self.tau = tau
        self.device = torch.device(device)
        self.policy = policy if policy is not None else TD3Policy(env_params=env_params, num_critics=2, device=device)
        self.policy.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.learning_rate) for critic in self.policy.critic.critics
        ]

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
                noise = utils.gaussian_noise(mean=0, std=0.1, shape=action.shape, device=self.device)
                action = action + noise
                # action = action.clamp(self.env_params.action_min, self.env_params.action_max)
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

        # Make collector and flatten the experience so the shape is (N*T, ...)
        collector = SequentialCollector(env=env, logger=logger)
        replay_buffer = ReplayBuffer(capacity=self.buffer_size, device=self.device)

        while num_steps < total_steps:
            # Update Schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)

            # Collect experience dictionary with shape (B, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.batch_size)
            num_steps += experience['state'].shape[0]

            # Store experience in replay buffer
            replay_buffer.add(experience)

            # Collect a minimum number of steps in the replay buffer before training
            if replay_buffer.get_size() < self.min_buffer_size:
                if show_progress:
                    progress_bar.update(current_step=num_steps, desc="Collecting initial experience...")
                continue

            actor_losses = []
            critics_losses = []
            for _ in range(self.gradient_steps):
                num_gradient_steps += 1

                # Sample a batch of experiences from the replay buffer
                batch = replay_buffer.sample(batch_size=self.mini_batch_size)

                # Compute current policy's action and target
                with torch.no_grad():
                    # Compute the policies next action with noise and clip to ensure it does not exceed action bounds
                    noise = utils.gaussian_noise(mean=0, std=self.policy_noise, shape=batch['action'].shape, device=self.device)
                    noise_clipped = noise.clamp(-self.noise_clip, self.noise_clip)
                    next_action = (self.policy.target_actor(batch['next_state'].float()) + noise_clipped) #.clamp(self.env_params.action_min, self.env_params.action_max)

                    # Compute the Q-Values for all the critics
                    next_q_values = self.policy.critic_target(batch['next_state'].float(), next_action)
                    next_q_values = torch.cat(next_q_values, dim=1)

                    # Use the minimum Q-Value across critics for the target
                    next_q_values = torch.min(next_q_values, dim=1, keepdim=True)[0] 

                    # Compute the target Q-Value
                    y = batch['reward'] + self.gamma * (1 - batch['done'].float()) * next_q_values

                # Perform gradient step on critics
                q_input = torch.cat([batch['state'].float(), batch['action']], dim=1)
                
                c_losses = []
                for i in range(self.policy.num_critics):
                    # Compute critics loss
                    q_i = self.policy.critic.critics[i](q_input.detach())
                    critic_loss = F.mse_loss(y, q_i)
                    c_losses.append(critic_loss.item())

                    self.critic_optimizers[i].zero_grad()
                    critic_loss.backward()
                    self.critic_optimizers[i].step()

                # Delayed policy update 
                if num_gradient_steps % self.delay_freq == 0:
                    # Compute actor loss
                    actor_loss = -self.policy.critic.forward_indexed(index=0, state=batch['state'], action=self.policy.actor(batch['state'])).mean()
                    actor_losses.append(actor_loss.item())

                    # Take a gradient step on the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Update target networks
                    utils.polyak_update(self.policy.target_actor, self.policy.actor, tau=self.tau)
                    for i in range(self.policy.num_critics):
                        utils.polyak_update(self.policy.critic_target.critics[i], self.policy.critic.critics[i], tau=self.tau)

            if show_progress:
                progress_bar.update(current_step=num_steps, desc=f"Episode Reward: {collector.previous_episode_reward:.2f}")

            if logger.should_log(num_steps):
                logger.log_scalar('actor_loss', np.mean(actor_losses), num_steps)
                for i in range(self.policy.num_critics):
                    logger.log_scalar(f'critic{i}_loss', np.mean(critics_losses[i]), num_steps)

            if evaluator is not None:
                evaluator.evaluate(agent=self.policy, num_steps=num_steps)

        if evaluator is not None:
            evaluator.close()