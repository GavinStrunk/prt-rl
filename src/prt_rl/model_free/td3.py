"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
"""
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List, Tuple, Dict
from prt_rl.agent import Agent
from prt_rl.common.policies import NeuralPolicy, RandomPolicy
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils

import copy
import numpy as np
from dataclasses import dataclass, asdict
from prt_rl.common.collectors import Collector
from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.components.heads import ContinuousHead, QValueHead


class Critic(nn.Module):
    def __init__(self, network: nn.Module, critic_head: nn.Module):
        super().__init__()
        self.network = network
        self.critic_head = critic_head

    def forward(self, obs, action):
        features = self.network(obs)
        q = self.critic_head(features, action)   # adjust signature to your head
        return q

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

class TD3Policy(NeuralPolicy):
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
                 network: nn.Module,
                 actor_head: ContinuousHead,
                 critic_head: QValueHead,
                 *,
                 action_min: Tensor,
                 action_max: Tensor,
                 num_critics: int = 2,
                 exploration_noise: float = 0.1,
                 critic_network: Optional[nn.Module] = None,
                 ) -> None:
        super().__init__()
        self.num_critics = num_critics
        self.action_min = action_min
        self.action_max = action_max
        self.exploration_noise = exploration_noise

        # Create an unified actor network and target actor network
        self.actor = nn.Sequential(network, actor_head)
        self.target_actor = copy.deepcopy(self.actor)

        # Create a separate critic network backbone by default if one is not provided
        critic_network = critic_network if critic_network is not None else copy.deepcopy(network)    

        # Create critics and target critics
        self.critics = nn.ModuleList()
        self.target_critics = nn.ModuleList()
        for _ in range(self.num_critics):
            critic = Critic(copy.deepcopy(critic_network), copy.deepcopy(critic_head))
            target_critic = copy.deepcopy(critic)

            self.critics.append(critic)
            self.target_critics.append(target_critic)

    def metadata(self):
        return {
            "num_critics": self.num_critics,
            "action_min": self.action_min,
            "action_max": self.action_max,
            "exploration_noise": self.exploration_noise,
        }

    @torch.no_grad()
    def act(self,
            obs: torch.Tensor,
            deterministic: bool = False
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        action = self.actor(obs)
        if not deterministic:
            # Add noise to the action for exploration
            noise = utils.gaussian_noise(mean=0, std=self.exploration_noise, shape=action.shape, device=self.device)
            action = action + noise
        
        # Ensure action is within bounds    
        action = action.clamp(self.action_min.to(self.device), self.action_max.to(self.device))
        return action, {}
    
    def target_actor_action(self, obs:Tensor, policy_noise: float, noise_clip: float, action_shape) -> torch.Tensor:
        """
        Compute the target actor's action with added noise for policy smoothing.
        Args:
            obs (torch.Tensor): The current observation of the environment.
            policy_noise (float): Standard deviation of noise added to the target policy's actions.
            noise_clip (float): Maximum absolute value of noise added to the target policy's actions.
            action_shape: Shape of the action tensor.
        Returns:
            torch.Tensor: The action computed by the target actor with added noise, clipped to action bounds.
        """
        # Generate additive Gaussian noise and clip it to the specified range
        noise = utils.gaussian_noise(mean=0, std=policy_noise, shape=action_shape, device=self.device)
        noise_clipped = noise.clamp(-noise_clip, noise_clip)

        # Get target actor action plus noise and clip to action bounds
        action = self.target_actor(obs) + noise_clipped
        action = action.clamp(self.action_min.to(self.device), self.action_max.to(self.device))
        return action
        
    def get_q_values(self,
                     obs: torch.Tensor,
                     action: torch.Tensor,
                     index: Optional[int] = None
                     ) -> torch.Tensor:
        """
        Get Q-values from all critics for the given state-action pairs.

        Args:
            obs (torch.Tensor): Current observation tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Tensor containing Q-values from all critics. Shape (B, C, 1) where C is the number of critics.
        """
        if index is None:
            q_values = [critic(obs, action) for critic in self.critics]
            q_values = torch.stack(q_values, dim=1)  # Shape (B, C, 1) where C is the number of critics
        else:
            q_values = self.critics[index](obs, action)
        return q_values
    
    def get_target_q_values(self,
                            obs: torch.Tensor,
                            action: torch.Tensor,
                            ) -> torch.Tensor:
        """
        Get target Q-values from all target critics for the given state-action pairs.

        Args:
            obs (torch.Tensor): Current observation tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            torch.Tensor: Tensor containing target Q-values from all critics. Shape (B, C, 1) where C is the number of critics.
        """
        q_values = [critic(obs, action) for critic in self.target_critics]
        q_values = torch.stack(q_values, dim=1)  # Shape (B, C, 1) where C is the number of critics
        return q_values
    

class TD3Agent(Agent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)

    This class implements the TD3 algorithm, which is an off-policy actor-critic algorithm for continuous action spaces.

    Args:
        policy (TD3Policy | None): Custom TD3 policy. If None, a default TD3 policy will be created.
        config (TD3Config): Configuration for the TD3 agent.
        device (str): Device to run the agent on ('cpu' or 'cuda'). Default is 'cpu'.
    """
    def __init__(self,
                 policy: TD3Policy,
                 config: TD3Config = TD3Config(),
                 *,
                 device: str = 'cpu',
                 ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)

        self.policy = policy
        self.policy.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.policy.critics.parameters(), lr=self.config.learning_rate)

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        """
        Perform an action based on the current state.

        Args:
            obs (torch.Tensor): The current observation of the environment.
            deterministic (bool): If True, the agent will select actions deterministically.

        Returns:
            torch.Tensor: The action to be taken.
        """
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
        logger = logger or Logger()
        evaluator = evaluator or Evaluator()

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        num_steps = 0
        num_gradient_steps = 0

        # Make collector and flatten the experience so the shape is (B, ...)
        collector = Collector(env=env, logger=logger, flatten=True)
        replay_buffer = ReplayBuffer(capacity=self.config.buffer_size, device=self.device)

        # Collect initial experience until replay buffer has enough samples for training with random policy
        random_policy = RandomPolicy(env.get_parameters())
        while replay_buffer.get_size() < self.config.min_buffer_size:
            experience = collector.collect_experience(policy=random_policy, num_steps=self.config.steps_per_batch)
            replay_buffer.add(experience)
            num_steps += experience['state'].shape[0]

            if show_progress:
                progress_bar.update(current_step=num_steps, desc="Collecting initial experience...")

        while num_steps < total_steps:
            self._update_schedulers(schedulers, num_steps)

            # Collect experience dictionary with shape (B, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.config.steps_per_batch)
            num_steps += experience['state'].shape[0]

            # Store experience in replay buffer
            replay_buffer.add(experience)

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
                    # Compute the policies next action with noise and clip to ensure it does not exceed action bounds - [B, A]
                    next_action = self.policy.target_actor_action(
                        obs=batch['next_state'],
                        policy_noise=self.config.policy_noise,
                        noise_clip=self.config.noise_clip,
                        action_shape=batch['action'].shape
                    )

                    # Compute the Q-Values for all the critics - shape (B, C, 1) -> (B, C)
                    next_q_values = self.policy.get_target_q_values(batch['next_state'], next_action).squeeze(-1) 

                    # Use the minimum Q-Value across critics for the target
                    next_q_values = torch.min(next_q_values, dim=1, keepdim=True)[0] 

                    # Compute the target Q-Value
                    y = batch['reward'] + self.config.gamma * (1 - batch['done'].float()) * next_q_values

                # Sum the losses across all critics
                qs = [self.policy.get_q_values(batch['state'].detach(), batch['action'].detach(), index=i) for i in range(self.policy.num_critics)]
                critic_loss = sum(F.mse_loss(y, q) for q in qs)
                critics_losses.append(critic_loss.item())

                # Take a gradient step on the critics
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Delayed policy update 
                if num_gradient_steps % self.config.delay_freq == 0:
                    # Compute actor loss
                    actor_loss = -self.policy.get_q_values(obs=batch['state'], action=self.policy.actor(batch['state']), index=0).mean()
                    actor_losses.append(actor_loss.item())

                    # Take a gradient step on the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Update target networks
                    utils.polyak_update(self.policy.target_actor, self.policy.actor, tau=self.config.tau)
                    for i in range(self.policy.num_critics):
                        utils.polyak_update(self.policy.target_critics[i], self.policy.critics[i], tau=self.config.tau)

            if show_progress:
                tracker = collector.get_metric_tracker()
                progress_bar.update(current_step=num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}"
                                                                f" Episode Length: {tracker.last_episode_length}"
                                                                f" Episode number: {tracker.episode_count}"
                                                                f" Actor Loss: {np.mean(actor_losses):.4f}"
                                                                )

            if logger.should_log(num_steps):
                logger.log_scalar('actor_loss', np.mean(actor_losses), num_steps)
                logger.log_scalar(f'critic_loss', critic_loss.detach().cpu().item(), num_steps)

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
            "algo": "TD3",
            "agent_format_version": 1,
            "config": asdict(self.config),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        torch.save(payload, path / "agent.pt")
        self.policy.save(path / "policy.pt")


    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "TD3Agent":
        """
        Loads the checkpoint and returns a fully-constructed TD3Agent.
        """
        p = Path(path)
        agent_meta = torch.load(p / "agent.pt", map_location=map_location, weights_only=False)

        if agent_meta.get("algo") != "TD3":
            raise ValueError(f"Checkpoint algo mismatch: expected TD3, got {agent_meta.get('algo')}")

        config = TD3Config(**agent_meta["config"])
        policy = TD3Policy.load(p / "policy.pt", map_location=map_location)

        agent = cls(
            policy=policy,
            config=config,
            device=str(map_location),
        )

        actor_opt_state = agent_meta["actor_optimizer_state_dict"]
        critic_opt_state = agent_meta["critic_optimizer_state_dict"]
        agent.actor_optimizer.load_state_dict(actor_opt_state)
        agent.critic_optimizer.load_state_dict(critic_opt_state)

        return agent         