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

        # self.env_encoder = EnvironmentEncoder(
        #     env_dim=(self.policy.env_params.observation_shape[0] + self.policy.env_params.action_shape[0]) * self.policy.time_steps,
        #     extrinsics_dim=self.policy.extrinsics_dim
        # ).to(device)

        self.adaptation_module = AdaptationModule(
            env_dim=(self.policy.env_params.observation_shape[0] + self.policy.env_params.action_shape[0]) * self.policy.time_steps,
            extrinsics_dim=self.policy.extrinsics_dim
        ).to(device)

        # Construct optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.adaptation_optimizer = torch.optim.Adam(self.adaptation_module.parameters(), lr=self.config.adapt_learning_rate)

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
        # Use the default PPO agent traing loop for Phase 1
        self.agent.train(
            env=env,
            total_steps=total_steps,
            schedulers=schedulers,
            logger=logger,
            evaluator=evaluator,
            show_progress=show_progress
        )

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