"""
Implementation of the Advantage Actor-Critic (A2C) algorithm.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List, Tuple, Dict
from prt_rl.agent import Agent
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.collectors import Collector
from prt_rl.common.buffers import RolloutBuffer
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils

import copy
from prt_rl.common.policies import NeuralPolicy
from prt_rl.common.components.heads import DistributionHead, ValueHead

@dataclass
class A2CConfig:
    """
    Configuration parameters for the A2C agent.
    
    Attributes:
        steps_per_batch (int): Number of steps to collect per training batch.
        mini_batch_size (int): Size of each mini-batch for optimization.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        entropy_coef (float): Coefficient for the entropy bonus.
        value_coef (float): Coefficient for the value loss.
        normalize_advantages (bool): Whether to normalize advantages.
    """
    steps_per_batch: int = 2048
    mini_batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = False

class A2CPolicy(NeuralPolicy):
    def __init__(self,
                 network: nn.Module,
                 actor_head: DistributionHead,
                 critic_head: ValueHead,
                 ) -> None:
        super().__init__()
        self.network = network
        self.actor_head = actor_head
        self.critic_head = critic_head
    
    @torch.no_grad()
    def act(self,
            obs: torch.Tensor,
            deterministic: bool = False
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns action + info dict.

        Info dict keys (typical):
          - "log_prob": (B,1)
          - "value":    (B,1)
        """        
        latent = self.network(obs)

        action, log_prob, _ = self.actor_head.sample(latent, deterministic=deterministic)
        value = self.critic_head(latent)

        return action, {"log_prob": log_prob, "value": value}
    
    def forward(self,
                obs: torch.Tensor,
                deterministic: bool = False
                ) -> torch.Tensor:
        """
        Convenience: treat the policy like a normal nn.Module that outputs actions.
        Collectors should call act() instead to get info dict.
        """        
        action, _ = self.act(obs, deterministic=deterministic)
        return action    
    
    def evaluate_actions(self,
                         obs: torch.Tensor,
                         action: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used during A2C optimization.

        Returns:
          value:    (B,1)
          log_prob: (B,1)
          entropy:  (B,1)
        """
        latent = self.network(obs)
        value = self.critic_head(latent)

        # Compute log probabilities and entropy for the entire action vector
        log_prob = self.actor_head.log_prob(latent, action)
        entropy = self.actor_head.entropy(latent)

        return value, log_prob, entropy

class A2CAgent(Agent):
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
                 *,
                 device: str = 'cpu',
                 ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)

        self.policy = policy
        self.policy.to(self.device)

        # Configure optimizers
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)   

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
        Train the A2C agent.

        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of steps to train for.
            schedulers (Optional[List[ParameterScheduler]]): Learning rate schedulers.
            logger (Optional[Logger]): Logger for training metrics.
            evaluator (Optional[Evaluator]): Evaluator for performance evaluation.
            show_progress (bool): If True, show a progress bar during training.
        """
        logger = logger or Logger()
        evaluator = evaluator or Evaluator()

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        num_steps = 0

        # Make collector and do not flatten the experience so the shape is (N, T, ...)
        collector = Collector(env=env, logger=logger, flatten=False)
        rollout_buffer = RolloutBuffer(capacity=self.config.steps_per_batch, device=self.device)

        while num_steps < total_steps:
            self._update_schedulers(schedulers=schedulers, step=num_steps)
            self._update_optimizer(self.optimizer, self.config.learning_rate)

            # Collect experience dictionary with shape (N, T, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.config.steps_per_batch)

            # Compute Advantages and Returns under the current policy
            experience = self._compute_gae(experience, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)

            num_steps += experience['state'].shape[0]

            # Add experience to the rollout buffer
            rollout_buffer.add(experience)

            # Optimization Loop
            policy_losses = []
            entropy_losses = []
            value_losses = []
            losses = []
            for batch in rollout_buffer.get_batches(batch_size=self.config.mini_batch_size):
                advantages = batch['advantages'].detach()
                returns = batch['returns'].detach()

                if self.config.normalize_advantages:
                    advantages = utils.normalize_advantages(advantages)

                # Get the log probability and entropy of the actions under the current policy
                new_value_est, new_log_prob, entropy = self.policy.evaluate_actions(batch['state'], batch['action'])
                
                policy_loss = -(new_log_prob * advantages).mean()

                # Compute entropy loss
                entropy_loss = -entropy.mean()

                # Compute the value loss function
                value_loss = F.mse_loss(new_value_est, returns)

                loss = policy_loss + self.config.entropy_coef*entropy_loss + self.config.value_coef * value_loss
                
                policy_losses.append(policy_loss.item())
                entropy_losses.append(entropy_loss.item())
                value_losses.append(value_loss.item())
                losses.append(loss.item())

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
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

            evaluator.evaluate(agent=self.policy, iteration=num_steps)

        evaluator.close()  

    def _save_impl(self, path: Path) -> None:
        """
        Writes a self-contained checkpoint directory.

        Layout:
          path/
            agent.pt
            policy.pt
        """
        path.mkdir(parents=True, exist_ok=True)

        payload = {
            "algo": "A2C",
            "agent_format_version": 1,
            "config": asdict(self.config),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, path / "agent.pt")
        self.policy.save(path / "policy.pt")


    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "A2CAgent":
        """
        Loads the checkpoint and returns a fully-constructed A2CAgent.
        """
        p = Path(path)
        agent_meta = torch.load(p / "agent.pt", map_location=map_location, weights_only=False)

        if agent_meta.get("algo") != "A2C":
            raise ValueError(f"Checkpoint algo mismatch: expected A2C, got {agent_meta.get('algo')}")

        config = A2CConfig(**agent_meta["config"])
        policy = A2CPolicy.load(p / "policy.pt", map_location=map_location)

        agent = cls(
            policy=policy,
            config=config,
            device=str(map_location),
        )

        opt_state = agent_meta["optimizer_state_dict"]
        agent.optimizer.load_state_dict(opt_state)

        return agent     