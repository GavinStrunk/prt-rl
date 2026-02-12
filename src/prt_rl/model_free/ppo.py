"""
Proximal Policy Optimization (PPO)

Reference:
[1] https://arxiv.org/abs/1707.06347
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, List, Tuple
from prt_rl.agent import Agent
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.collectors import Collector
from prt_rl.common.buffers import RolloutBuffer
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils

import prt_rl.common.policies as pmod
from prt_rl.common.components.heads.interface import DistributionHead
from prt_rl.common.components.heads import ValueHead

# @todo Add support for KL stopping
# @todo Add support for value clipping

# Define the Algorithm config dataclass
@dataclass
class PPOConfig:
    """
    Configuration for the PPO agent.

    Args:
        steps_per_batch (int): Number of steps to collect per batch.
        mini_batch_size (int): Size of mini-batches for optimization.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Clipping parameter for PPO.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
        entropy_coef (float): Coefficient for the entropy term in the loss function.
        value_coef (float): Coefficient for the value loss term in the loss function.
        num_optim_steps (int): Number of optimization steps per batch.
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
    num_optim_steps: int = 10
    normalize_advantages: bool = False

# Define the Policy Interface
class PPOPolicy(pmod.Policy):
    """
    PPOPolicy is a policy that combines an actor and a critic network. It can optionally use an encoder network to process the input state before passing it to the actor and critic heads.

    The PPOPolicy is a combination of a DistributionPolicy for the actor and a ValueCritic for the critic. It can handle both discrete and continuous action spaces.
    
    The architecture of the policy is as follows:
        - Encoder Network (optional): Processes the input state.
        - Actor Head: Computes actions based on the latent state.
        - Critic Head: Computes the value for the given state.

    .. image:: /_static/actorcriticpolicy.png
        :alt: ActorCriticPolicy Architecture
        :width: 100%
        :align: center

    Args:
        env_params (EnvParams): Environment parameters.
        encoder (BaseEncoder | None): Encoder network to process the input state. If None, the input state is used directly.
        actor (DistributionPolicy | None): Actor network to compute actions. If None, a default DistributionPolicy is created.
        critic (ValueCritic | None): Critic network to compute values. If None, a default ValueCritic is created.
        share_encoder (bool): If True, share the encoder between actor and critic. Default is False.
    """
    def __init__(self,
                *,
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
            ) -> Tuple[torch.Tensor, pmod.InfoDict]:
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
        Used during PPO optimization.

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
    
    def get_state_value(self,
                         obs: torch.Tensor,
                         ) -> torch.Tensor:
        """
        Returns the state value for the given observation.
        
        Args:
          obs:      (B, obs_dim)
        Returns:
          value:    (B,1)
        """
        latent = self.network(obs)
        value = self.critic_head(latent)
        return value

# Make the Agent
class PPOAgent(Agent):
    """
    Proximal Policy Optimization (PPO)

    Args:
        policy (PPOPolicy): Policy to use.
        config (PPOConfig): Configuration for the PPO agent.
        device (str): Device to run the computations on ('cpu' or 'cuda').
    """
    def __init__(self,
                 policy: PPOPolicy,
                 config: PPOConfig = PPOConfig(),
                 *,
                 device: str = 'cpu',
                 ) -> None:
        self.config = config
        self.policy = policy.to(device)

        super().__init__(device=device)

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
        Train the PPO agent.

        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of steps to train for.
            schedulers (Optional[List[ParameterScheduler]]): Learning rate schedulers.
            logger (Optional[Logger]): Logger for training metrics.
            evaluator (Optional[Evaluator]): Evaluator for performance evaluation.
            show_progress (bool): If True, show a progress bar during training.
        """
        logger = logger or Logger()

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        num_steps = 0

        # Make collector and do not flatten the experience so the shape is (T, N, ...)
        collector = Collector(env=env, logger=logger, flatten=False)
        rollout_buffer = RolloutBuffer(capacity=self.config.steps_per_batch, device=self.device)

        while num_steps < total_steps:
            # Update Schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)

            # Collect experience dictionary with shape (T, N, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.config.steps_per_batch)

            # Compute Advantages and Returns under the current policy
            advantages, returns = utils.generalized_advantage_estimates(
                rewards=experience['reward'],
                values=experience['value'],
                dones=experience['done'],
                last_values=experience['last_value_est'],
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda
            )
            
            experience['advantages'] = advantages
            experience['returns'] = returns

            # Flatten the experience batch (T, N, ...) -> (T*N, ...) and remove the last_value_est key because we don't need it anymore
            experience = {k: v.reshape(-1, *v.shape[2:]) for k, v in experience.items() if k != 'last_value_est'}

            # Update the total number of steps collected so far
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
                    # Treat the previous policy's log probabilities as constant, as well as the advantages and returns
                    old_log_prob = batch['log_prob']
                    advantages = batch['advantages']
                    returns = batch['returns']

                    if self.config.normalize_advantages:
                        advantages = utils.normalize_advantages(advantages)

                    # Get the log probability and entropy of the actions under the current policy
                    new_value_est, new_log_prob, entropy = self.policy.evaluate_actions(batch['state'], batch['action'])
                    
                    # Ratio between new and old policy
                    ratio = torch.exp(new_log_prob - old_log_prob)

                    # Clipped surrogate loss
                    clip_loss = advantages * ratio
                    clip_loss2 = advantages * torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
                    clip_loss = -torch.min(clip_loss, clip_loss2).mean()

                    # Compute entropy loss
                    entropy_loss = -entropy.mean()

                    # Compute the value loss function
                    value_loss = F.mse_loss(new_value_est, returns)

                    # Compute total clipped PPO loss
                    loss = clip_loss + self.config.entropy_coef*entropy_loss + self.config.value_coef * value_loss
                    
                    clip_losses.append(clip_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    value_losses.append(value_loss.item())
                    losses.append(loss.item())

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
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
            "algo": "PPO",
            "agent_format_version": 1,
            "config": asdict(self.config),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, path / "agent.pt")
        self.policy.save(path / "policy.pt")


    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PPOAgent":
        """
        Loads the checkpoint and returns a fully-constructed PPOAgent.
        """
        p = Path(path)
        agent_meta = torch.load(p / "agent.pt", map_location=map_location, weights_only=False)

        if agent_meta.get("algo") != "PPO":
            raise ValueError(f"Checkpoint algo mismatch: expected PPO, got {agent_meta.get('algo')}")

        config = PPOConfig(**agent_meta["config"])
        policy = PPOPolicy.load(p / "policy.pt", map_location=map_location)

        agent = cls(
            policy=policy,
            config=config,
            device=str(map_location),
        )

        opt_state = agent_meta["optimizer_state_dict"]
        agent.optimizer.load_state_dict(opt_state)

        return agent    



