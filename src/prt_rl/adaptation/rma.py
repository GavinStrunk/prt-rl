"""
Rapid Motor Adaptation
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
from prt_rl.agent import Agent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator

import torch.nn.functional as F
from prt_rl.common.policies import NeuralPolicy
import prt_rl.common.utils as utils
from prt_rl.model_free.ppo import PPOConfig, PPOAgent, PPOPolicy


@dataclass
class RMAConfig:
    extrinsics_dim: int
    ppo_config: PPOConfig = PPOConfig()
    adapt_learning_rate: float = 5e-4
    adapt_batch_size: int = 80_000
    adapt_mini_batch_size: int = 4


class RMAPolicy(NeuralPolicy):
    """
    Rapid Motor Adaptation (RMA) Policy.
    
    The policy takes the current state, previous action, and environment encoding as input.
    """
    def __init__(self, 
                 ppo_policy: PPOPolicy,
                 adaptation_module: nn.Module,
                 extrinsics_dim: int
                 ) -> None:
        super().__init__()
        self.ppo_policy = ppo_policy
        self.adaptation_module = adaptation_module
        self.extrinsics_dim = extrinsics_dim

    @torch.no_grad()
    def act(self,
            obs: torch.Tensor,
            deterministic: bool = False
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Observation is 
        state = obs[:, :-self.extrinsics_dim]
        e_t = obs[:, -self.extrinsics_dim:]
        pass



class RMAAgent(Agent):
    def __init__(self,
                 policy: RMAPolicy,
                 config: RMAConfig,
                 *,
                 env_encoder: nn.Module,
                 device: str = 'cpu'
                 ) -> None:
        super().__init__()
        self.config = config
        self.policy = policy.to(device)
        self.device = device
        self.env_encoder = env_encoder.to(device)

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        action, _ = self.policy.act(obs, deterministic=deterministic)
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
        Train the RMA agent.
        
        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of environment steps to train for.
            schedulers (List[ParameterScheduler] | None): List of parameter schedulers to update during training.
            logger (Logger | None): Logger for logging training metrics. If None, a default logger will be created.
            evaluator (Evaluator | None): Evaluator for periodic evaluation during training.
            show_progress (bool): If True, display a progress bar during training.
        """
        # Make a local policy to train with PPO
        class Phase1Policy(PPOPolicy):
            def __init__(self, extrinisic_dim: int, env_encoder: nn.Module, ppo_policy: PPOPolicy) -> None:
                super().__init__()
                self.extrinisic_dim = extrinisic_dim
                self.env_encoder = env_encoder
                self.ppo_policy = ppo_policy
            @torch.no_grad()
            def act(self,
                    obs: torch.Tensor,
                    deterministic: bool = False
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                # Observation is [s_t, a_{t-1}, e_t], split out the extrinsics
                state_t = obs[:, :-self.extrinisic_dim]
                e_t = obs[:, -self.extrinisic_dim:]

                # Compute latent environment factor encoding
                z_t = self.env_encoder(e_t)

                # Augment the base policy state to become [s_t, a_{t-1}, z_t]
                state_ext = torch.cat([state_t, z_t], dim=-1)

                action, info = self.ppo_policy.act(state_ext, deterministic=deterministic)

                # add z_t to info dictionary
                info["z_t"] = z_t
                return action, info
        
        phase1_policy = Phase1Policy(extrinisic_dim=self.config.extrinsics_dim, env_encoder=self.env_encoder, ppo_policy=self.policy.ppo_policy)
        ppo_agent = PPOAgent(policy=phase1_policy, config=self.config.ppo_config, device=self.device)

        ppo_agent.train(
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