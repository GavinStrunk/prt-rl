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
from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.policies import StateActionCritic

class TD3Policy:
    """
    Placeholder for TD3 policy class.
    This should be implemented with the actual policy logic.

    """
    def __init__(self, 
                 env_params, 
                 num_critics: int = 2,
                 device='cpu'
                 ) -> None:
        self.env_params = env_params
        self.num_critics = num_critics
        self.device = torch.device(device)

        self.critic = StateActionCritic(env_params=env_params, num_critics=num_critics)
        self.critic.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.to(self.device)

class TD3(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)

    """
    def __init__(self,
                 env_params: EnvParams,
                 policy = None,
                 batch_size: int = 256,
                 min_num_steps: int = 1000,
                 gradient_steps: int = 1,
                 mini_batch_size: int = 256,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 delay_freq: int = 2,
                 tau: float = 0.005,
                 device: str = 'cpu',
                 ) -> None:
        super().__init__(policy=policy)
        self.env_params = env_params
        self.batch_size = batch_size
        self.min_num_steps = min_num_steps
        self.gradient_steps = gradient_steps
        self.mini_batch_size = mini_batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.delay_freq = delay_freq
        self.tau = tau
        self.device = torch.device(device)
        self.policy.to(self.device)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform an action based on the current state.

        Args:
            state: The current state of the environment.

        Returns:
            The action to be taken.
        """
        with torch.no_grad():
            return self.policy(state)
        
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              logging_freq: int = 1000,
              evaluator: Evaluator = Evaluator(),
              eval_freq: int = 1000,
              ) -> None:
        """
        Update the agent's knowledge based on the action taken and the received reward.
        This method should implement the TD3 training loop.

        Args:
            env: The environment to interact with.
            total_steps: Total number of steps to train the agent.
            schedulers: Optional list of parameter schedulers.
            logger: Optional logger for logging training progress.
            logging_freq: Frequency of logging.
            evaluator: Evaluator for evaluating the agent's performance.
            eval_freq: Frequency of evaluation.
        """
        logger = logger or Logger.create('blank')
        progress_bar = ProgressBar(total_steps=total_steps)
        num_steps = 0
        num_gradient_steps = 0

        # Make collector and do not flatten the experience so the shape is (N, T, ...)
        collector = ParallelCollector(env=env, logger=logger, logging_freq=logging_freq, flatten=False)
        replay_buffer = ReplayBuffer(capacity=100000, device=self.device)

        critic_losses = []
        actor_losses = []
        while num_steps < total_steps:
            # Update Schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)

            # Collect experience dictionary with shape (T, N, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.batch_size)
            num_steps += experience['state'].shape[0] * experience['state'].shape[1]

            # Store experience in replay buffer
            replay_buffer.add(experience)

            # Collect a minimum number of steps in the replay buffer before training
            if replay_buffer.get_size() < self.min_num_steps:
                progress_bar.update(current_step=num_steps, desc="Collecting initial experience...")
                continue

            for _ in range(self.gradient_steps):
                num_gradient_steps += 1

                # Sample a batch of experiences from the replay buffer
                batch = replay_buffer.sample(batch_size=self.mini_batch_size)

                # Compute current policy's action and target
                with torch.no_grad():
                    # Compute the policies next action with noise and clip to ensure it does not exceed action bounds
                    noise = utils.gaussian_noise(mean=0, std=self.policy_noise, size=batch['action'].shape)
                    noise_clipped = noise.clamp(-self.noise_clip, self.noise_clip)
                    next_action = (self.target_actor(batch['next_state']) + noise_clipped).clamp(self.env_params.action_min, self.env_params.action_max)

                    # Compute the Q-Values for all the critics
                    next_q_values = self.target_critic.evaluate_actions(batch['next_state'], next_action)

                    # Use the minimum Q-Value across critics for the target
                    next_q_values = torch.min(next_q_values, dim=1, keepdim=True)[0] 

                    # Compute the target Q-Value
                    y = batch['reward'] + self.env_params.gamma * (1 - batch['done'].float()) * next_q_values

                # Perform gradient step on critics
                current_q_values = self.policy.evaluate_actions(batch['state'], batch['action'])

                # Compute critics loss
                critic_loss = F.mse_loss(y, current_q_values)
                critic_losses.append(critic_loss.item())

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Delayed policy update 
                if num_gradient_steps % self.delay_freq == 0:
                    # Compute actor loss
                    actor_loss = -self.critic1(batch['state'], self.actor(batch['state'])).mean()
                    actor_losses.append(actor_loss.item())

                    # Take a gradient step on the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Update target networks
                    utils.polyak_update(self.target_actor, self.actor, tau=self.tau)
                    utils.polyak_update(self.target_critic1, self.critic1, tau=self.tau)
                    utils.polyak_update(self.target_critic2, self.critic2, tau=self.tau)