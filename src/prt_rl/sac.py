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
import prt_rl.common.distributions as dist
import prt_rl.common.utils as utils

@dataclass
class SACConfig:
    """
    Hyperparameter configuration for the SAC agent.
    
    Args:
        buffer_size (int): Size of the replay buffer.
        min_buffer_size (int): Minimum number of transitions in the replay buffer before training starts.
        steps_per_batch (int): Number of steps to collect per training batch.
        mini_batch_size (int): Size of the mini-batch sampled from the replay buffer for training.
        gradient_steps (int): Number of gradient update steps to perform after each batch of experience is collected.
        learning_rate (float): Learning rate for the optimizers.
        tau (float): Soft update coefficient for the target networks.
        gamma (float): Discount factor for future rewards.
        entropy_coeff (float | None): Initial value for the entropy coefficient, alpha. If None, it will be learned.
        target_entropy (float | None): Target entropy for the policy. If None, it will be set to -action_dim.
        use_log_entropy (bool): If True, optimize the log of the entropy coefficient, else optimize the coefficient directly.
        reward_scale (float): Scaling factor for rewards.
    """
    buffer_size: int = 1000000
    min_buffer_size: int = 100
    steps_per_batch: int = 1
    mini_batch_size: int = 256
    gradient_steps: int = 1
    learning_rate: float = 3e-4
    tau: float = 0.005
    gamma: float = 0.99
    entropy_coeff: Optional[float] = None
    target_entropy: Optional[float] = None
    use_log_entropy: bool = True
    reward_scale: float = 1.0
    
class SACPolicy(BasePolicy):
    """
    Soft Actor-Critic (SAC) policy class.

    The default actor is a DistributionPolicy with a TanhGaussian distribution,
    and the default critic is a StateActionCritic with 2 critics.
    
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

        self.actor = actor if actor is not None else DistributionPolicy(env_params=env_params, policy_kwargs={'network_arch': [256, 256]}, distribution=dist.TanhGaussian)
        self.actor.to(self.device)

        self.critic = critic if critic is not None else StateActionCritic(env_params=env_params, num_critics=num_critics)
        self.critic.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.to(self.device)

        # Compute the scaling and bias for rescaling the actions
        amax = self.env_params.get_action_max_tensor().to(self.device)
        amin = self.env_params.get_action_min_tensor().to(self.device)
        self.action_scale = (amax - amin) / 2
        self.action_bias = (amax + amin) / 2

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
        # Get the action from the Squashed Gaussian policy which is in the range [-1, 1]
        action = self.actor(state, deterministic=deterministic)
        action = action * self.action_scale + self.action_bias
        return action

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

        # Rescale the action to the environment's action space
        action = action * self.action_scale + self.action_bias  

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
        
        # Construct a default policy is one is not provided
        self.policy = policy if policy is not None else SACPolicy(env_params=env_params, device=device) 
        self.policy.to(self.device)

        # Initialize the entropy coefficient and target
        if self.config.target_entropy is None:
            self.target_entropy = -float(env_params.action_len)
        else:
            self.target_entropy = self.config.target_entropy
        
        if self.config.entropy_coeff is None:
            if self.config.use_log_entropy:
                self.entropy_coeff = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
            else:
                self.entropy_coeff = torch.tensor(0.0, requires_grad=True, device=self.device)

            self.entropy_optimizer = torch.optim.Adam([self.entropy_coeff], lr=self.config.learning_rate)
        else:
            self.entropy_coeff = torch.tensor(self.config.entropy_coeff, device=self.device)
            self.entropy_optimizer = None

        # Configure the optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.config.learning_rate) for critic in self.policy.critic.critics
        ]

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

            # Apply reward scaling
            experience['reward'] = experience['reward'] * self.config.reward_scale

            # Add experience to the replay buffer
            replay_buffer.add(experience)

            # Collect a minimum number of steps in the replay buffer before training
            if replay_buffer.get_size() < self.config.min_buffer_size:
                if show_progress:
                    progress_bar.update(current_step=num_steps, desc="Collecting initial experience...")
                continue            

            actor_losses = []
            critics_losses = []
            entropy_losses = []
            for _ in range(self.config.gradient_steps):
                # Sample a mini-batch from the replay buffer
                mini_batch = replay_buffer.sample(batch_size=self.config.mini_batch_size)

                # Compute the current policy's action and log probability
                current_action, _, current_log_prob = self.policy.predict(mini_batch['state'])

                # Entropy coefficient optimization
                if self.config.use_log_entropy:
                    entropy_coeff = torch.exp(self.entropy_coeff.detach())
                else:
                    entropy_coeff = self.entropy_coeff

                if self.entropy_optimizer is not None:
                    entropy_loss = -(self.entropy_coeff * (current_log_prob + self.target_entropy).detach()).mean()
                    entropy_losses.append(entropy_loss.item())

                    self.entropy_optimizer.zero_grad()
                    entropy_loss.backward()
                    self.entropy_optimizer.step()

                # Compute the target values from the current policy
                with torch.no_grad():
                    # Select next action based on current policy
                    next_action, _, next_log_prob = self.policy.predict(mini_batch['next_state'])

                    # Compute the Q-values for all critics using target networks
                    next_q_values = self.policy.get_target_q_values(state=mini_batch['next_state'], action=next_action).squeeze(-1)
                    next_q_values = torch.min(next_q_values, dim=1, keepdim=True)[0]

                    # Add the entropy term to the target Q-values
                    next_q_values += -entropy_coeff * next_log_prob

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
                actor_loss = (entropy_coeff * current_log_prob - q_values_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update target critic networks
                for i in range(self.policy.num_critics):
                    utils.polyak_update(self.policy.critic_target.critics[i], self.policy.critic.critics[i], tau=self.config.tau)   

            if show_progress:
                tracker = collector.get_metric_tracker()
                progress_bar.update(current_step=num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}"
                                                                f" Episode Length: {tracker.last_episode_length}"
                                                                f" Episode number: {tracker.episode_count}"
                                                                f" Actor Loss: {np.mean(actor_losses):.4f}"
                                                                f" Entropy Coef: {entropy_coeff.item():.2f}"
                                                                )

            if logger.should_log(num_steps):
                logger.log_scalar('actor_loss', np.mean(actor_losses), num_steps)
                logger.log_scalar('entropy_loss', np.mean(entropy_losses), num_steps)
                logger.log_scalar('entropy_coeff', entropy_coeff.item(), num_steps)
                for i in range(self.policy.num_critics):
                    logger.log_scalar(f'critic{i}_loss', np.mean(critics_losses[i]), num_steps)

            if evaluator is not None:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)

        if evaluator is not None:
            evaluator.close()                                 
