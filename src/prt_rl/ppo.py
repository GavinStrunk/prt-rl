"""
Proximal Policy Optimization (PPO)

Reference:
[1] https://arxiv.org/abs/1707.06347
"""
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, List, Literal, Tuple, Union
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.buffers import RolloutBuffer
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils

import prt_rl.common.policies as pmod
import prt_rl.common.policies.encoders as enc
import prt_rl.common.policies.backbones as back
import prt_rl.common.policies.heads as heads


# 1) Define the Algorithm config dataclass
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

# 2) Define the PolicySpec dataclass
# @todo convert this to a HeadSpec and defined one for each head type
@dataclass
class PPOHeadSpec:
    distribution: Literal["auto", "categorical", "gaussian"] = "auto"
    log_std_init: float = -0.5
    min_log_std: float = -20.0
    max_log_std: float = 2.0

@dataclass
class PPOPolicySpec:
    """
    Describes how to build the PPO compliant policy.
    """
    encoder: enc.EncoderSpec = field(default_factory=enc.IdentityEncoderSpec)
    encoder_options: enc.EncoderOptions = field(default_factory=enc.EncoderOptions)
    backbone: back.BackboneSpec = field(default_factory=back.MLPBackboneSpec)
    heads: PPOHeadSpec = field(default_factory=PPOHeadSpec)
    share_backbone: bool = False

# 3) Define the Policy class as an extension of PolicyModule
class PPOPolicy(pmod.PolicyModule):
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
                encoder: nn.Module,
                backbone: nn.Module,
                actor_head: nn.Module,
                value_head: nn.Module,
                 ) -> None:
        super().__init__()
        self.encoder = encoder      
        self.backbone = backbone
        self.actor_head = actor_head
        self.critic_head = value_head

    def _latent(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent representation of the input observation.

        Args:
            obs (torch.Tensor): Input observation tensor.

        Returns:
            torch.Tensor: Latent representation tensor.
        """
        # Run state through encoder if available
        latent_state = self.encoder(obs) if self.encoder is not None else obs

        # Pass latent state through base network
        latent_state = self.backbone(latent_state)

        return latent_state
    
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
        latent = self._latent(obs)

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
        latent = self._latent(obs)
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
        latent = self._latent(obs)
        value = self.critic_head(latent)
        return value
    

# 4) Make a PolicyFactory to build the policy from the spec
class PPOPolicyFactory(pmod.PolicyFactory[PPOPolicySpec, PPOPolicy]):

    def make(self, env_params: EnvParams, spec: PPOPolicySpec) -> PPOPolicy:
        # Build Encoder
        encoder = enc.build_encoder(env_params, spec.encoder, spec.encoder_options)

        # Build Backbone
        backbone = back.build_backbone(encoder.latent_dim, spec.backbone)
        latent_dim = backbone.latent_dim

        # Choose distribution/head type
        dist = spec.heads.distribution
        if dist == "auto":
            dist = "categorical" if not env_params.action_continuous else "gaussian"

        # Create a Categorical Actor head for discrete actions
        if not env_params.action_continuous:
            num_actions = int(env_params.action_max - env_params.action_min + 1)

            if dist != "categorical":
                raise ValueError(f"Discrete env requires categorical policy; got distribution={dist}")
            
            actor = heads.CategoricalHead(latent_dim, num_actions)

        else:
            if dist not in ["gaussian"]:
                raise ValueError(f"Continuous env requires a continuous actor head; got distribution={dist}")
            
            actor = heads.GaussianHead(latent_dim, env_params.action_len)
        
        critic = heads.ValueHead(latent_dim)
        return PPOPolicy(
            encoder=encoder,
            backbone=backbone,
            actor_head=actor,
            value_head=critic,
        )

    def save(self, env_params: EnvParams, spec: PPOPolicySpec, policy: PPOPolicy, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        payload = {
            "env_params": asdict(env_params),
            "spec": asdict(spec),
            "format_version": 1,
        }
        (p / "spec.json").write_text(json.dumps(payload, indent=2))
        torch.save(policy.state_dict(), p / "weights.pt")

    def load(
        self,
        path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> Tuple[EnvParams, PPOPolicySpec, PPOPolicy]:
        p = Path(path)
        payload = json.loads((p / "spec.json").read_text())
        env_params = EnvParams(**payload["env_params"])
        spec = PPOPolicySpec(**payload["spec"])
        policy = self.make(env_params, spec)
        sd = torch.load(p / "weights.pt", map_location=map_location)
        policy.load_state_dict(sd, strict=strict)
        return env_params, spec, policy


# 5) Make the Agent
class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO)

    Args:
        policy (ActorCriticPolicy | None): Policy to use. If None, a default ActorCriticPolicy will be created.
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
        device (str): Device to run the computations on ('cpu' or 'cuda').
    """
    def __init__(self,
                 env_params: EnvParams,
                 policy_spec: PPOPolicySpec,
                 *,
                 config: PPOConfig = PPOConfig(),
                 device: str = 'cpu',
                 ) -> None:
        self.env_params = env_params
        self.policy_spec = policy_spec
        self.config = config
        policy = PPOPolicyFactory().make(env_params, policy_spec).to(device)

        super().__init__(policy=policy, device=device)

        # Configure optimizers
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
    
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

            experience['advantages'] = advantages
            experience['returns'] = returns

            # Flatten the experience batch (N, T, ...) -> (N*T, ...) and remove the last_value_est key because we don't need it anymore
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
                    old_log_prob = batch['log_prob'].detach()
                    advantages = batch['advantages'].detach()
                    returns = batch['returns'].detach()

                    # Get the log probability and entropy of the actions under the current policy
                    new_log_prob, entropy = self.policy.evaluate_actions(batch['state'], batch['action'])
                    
                    # Ratio between new and old policy
                    ratio = torch.exp(new_log_prob - old_log_prob)

                    # Clipped surrogate loss
                    clip_loss = advantages * ratio
                    clip_loss2 = advantages * torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
                    clip_loss = -torch.min(clip_loss, clip_loss2).mean()

                    # Compute entropy loss
                    entropy_loss = -entropy.mean()

                    # Compute the value loss function
                    new_value_est = self.policy.get_state_value(batch['state'])
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
            agent.json
            policy/
              spec.json
              weights.pt
            optimizer.pt
        """
        # Create the path if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Agent metadata/config (lets you sanity-check / migrate later)
        agent_meta = {
            "algo": "PPO",
            "agent_format_version": 1,
            "config": asdict(self.config),
        }
        agent_meta_path = path / "agent.json"
        agent_meta_path.write_text(json.dumps(agent_meta, indent=2))

        # Policy (delegated to factory)
        factory = PPOPolicyFactory()
        factory.save(self.env_params, self.policy_spec, self.policy, path / "policy")

        # Optimizer
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")


    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PPOAgent":
        """
        Loads the checkpoint and returns a fully-constructed PPOAgent.

        Note: This assumes the checkpoint was produced by PPOAgent._save_impl.
        """
        p = Path(path)
        agent_meta_path = p / "agent.json"
        agent_meta = json.loads(agent_meta_path.read_text())

        if agent_meta.get("algo") != "PPO":
            raise ValueError(f"Checkpoint algo mismatch: expected PPO, got {agent_meta.get('algo')}")

        config = PPOConfig(**agent_meta["config"])

        factory = PPOPolicyFactory()
        env_params, policy_spec, policy = factory.load(p / "policy", map_location=map_location)

        agent = cls(
            env_params=env_params,
            policy_spec=policy_spec,
            config=config,
            device=str(map_location),
        )

        # Update the policy with the saved policy parameters
        agent.policy = policy

        opt_state = torch.load(p / "optimizer.pt", map_location=map_location)
        agent.optimizer.load_state_dict(opt_state)

        return agent    




