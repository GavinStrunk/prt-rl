"""
Soft Actor-Critic (SAC)
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import itertools
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from prt_rl.agent import Agent
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.policies import NeuralPolicy, RandomPolicy
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator

import copy
import numpy as np
from prt_rl.common.collectors import Collector
from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.components.heads import DistributionHead, QValueHead
from prt_rl.common.components.networks import QCritic
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
        target_entropy (float | None): Target entropy for the policy. A reasonable default is -action_dim.
        use_log_entropy (bool): If True, optimize the log of the entropy coefficient, else optimize the coefficient directly.
        reward_scale (float): Scaling factor for rewards.
    """
    target_entropy: float
    buffer_size: int = 1000000
    min_buffer_size: int = 100
    steps_per_batch: int = 1
    mini_batch_size: int = 256
    gradient_steps: int = 1
    learning_rate: float = 3e-4
    tau: float = 0.005
    gamma: float = 0.99
    entropy_coeff: Optional[float] = None
    use_log_entropy: bool = True
    reward_scale: float = 1.0
    
class SACPolicy(NeuralPolicy):
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
                network: nn.Module,
                actor_head: DistributionHead,
                critic_head: QValueHead,
                *,
                action_min: Tensor,
                action_max: Tensor,
                num_critics: int = 2,
                critic_network: Optional[nn.Module] = None,
                ) -> None:
        super().__init__()
        self.num_critics = num_critics
        self.action_min = action_min
        self.action_max = action_max
        self.action_scale = (action_max - action_min) / 2
        self.action_bias = (action_max + action_min) / 2
        self.actor_network = network
        self.actor_head = actor_head

        # Create a separate critic network backbone by default if one is not provided
        critic_network = critic_network if critic_network is not None else copy.deepcopy(network)    

        # Create critics and target critics
        self.critics = nn.ModuleList()
        self.target_critics = nn.ModuleList()
        for _ in range(self.num_critics):
            critic = QCritic(copy.deepcopy(critic_network), copy.deepcopy(critic_head))
            target_critic = copy.deepcopy(critic)

            self.critics.append(critic)
            self.target_critics.append(target_critic)

    def metadata(self):
        return {
            "num_critics": self.num_critics,
            "action_min": self.action_min,
            "action_max": self.action_max,
        }
    
    @torch.no_grad()
    def act(self,
            obs: torch.Tensor,
            deterministic: bool = False
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict the action based on the current state.

        Args:
            state (torch.Tensor): Current state tensor.
            deterministic (bool): If True, choose the action deterministically. Default is False.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the chosen action, value estimate, and action log probability.
                - action (torch.Tensor): Tensor with the chosen action. Shape (B, action_dim)
                - log_prob (torch.Tensor): None
        """
        latent = self.actor_network(obs)
        action, log_prob, _ = self.actor_head.sample(latent, deterministic=deterministic) 

        # Rescale the action to the environment's action space
        action = action * self.action_scale.to(obs.device) + self.action_bias.to(obs.device)

        return action, {"log_prob": log_prob}
    
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


class SACAgent(Agent):
    """
    Soft Actor-Critic (SAC) agent.

    Args:
        policy (SACPolicy | None): Policy to use. If None, a default SACPolicy will be created.
        config (SACConfig): Configuration for the SAC agent.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').

    References:
        [1] https://arxiv.org/pdf/1812.05905    
    """
    def __init__(self, 
                 policy: SACPolicy,
                 config: SACConfig, 
                 *,
                 device: str = 'cpu'
                 ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        
        # Construct a default policy is one is not provided
        self.policy = policy 
        self.policy.to(self.device)

        # Initialize the entropy coefficient and target
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
        self.actor_optimizer = torch.optim.Adam(itertools.chain(self.policy.actor_network.parameters(), self.policy.actor_head.parameters()), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.policy.critics.parameters(), lr=self.config.learning_rate)


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
        Train the SAC agent.
        
        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of environment steps to train for.
            schedulers (List[ParameterScheduler] | None): List of parameter schedulers to update during training.
            logger (Logger | None): Logger for logging training metrics. If None, a default logger will be created.
            evaluator (Evaluator | None): Evaluator for periodic evaluation during training.
            show_progress (bool): If True, display a progress bar during training.
        """
        logger = logger or Logger()
        evaluator = evaluator or Evaluator()

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        num_steps = 0

        collector = Collector(env=env, logger=logger, flatten=True)   
        replay_buffer = ReplayBuffer(capacity=self.config.buffer_size, device=self.device)

        # Collect initial experience until replay buffer has enough samples for training with policy so we have log probabilities 
        while replay_buffer.get_size() < self.config.min_buffer_size:
            experience = collector.collect_experience(policy=self.policy, num_steps=self.config.steps_per_batch)

            # Apply reward scaling
            experience['reward'] = experience['reward'] * self.config.reward_scale  

            replay_buffer.add(experience)
            num_steps += experience['state'].shape[0]

            if show_progress:
                progress_bar.update(current_step=num_steps, desc="Collecting initial experience...")        

        while num_steps < total_steps:
            self._update_schedulers(schedulers=schedulers, step=num_steps) 

            # Collect experience dictionary with shape (B, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.config.steps_per_batch, bootstrap=False)
            num_steps += experience['state'].shape[0]

            # Apply reward scaling
            experience['reward'] = experience['reward'] * self.config.reward_scale

            # Add experience to the replay buffer
            replay_buffer.add(experience)

            actor_losses = []
            critics_losses = []
            entropy_losses = []
            for _ in range(self.config.gradient_steps):
                # Sample a mini-batch from the replay buffer
                mini_batch = replay_buffer.sample(batch_size=self.config.mini_batch_size)

                # Compute the current policy's action and log probability
                current_action, action_info = self.policy.act(mini_batch['state'])
                current_log_prob = action_info['log_prob']

                # Entropy coefficient optimization
                if self.config.use_log_entropy:
                    entropy_coeff = torch.exp(self.entropy_coeff.detach())
                else:
                    entropy_coeff = self.entropy_coeff

                if self.entropy_optimizer is not None:
                    entropy_loss = -(self.entropy_coeff * (current_log_prob + self.config.target_entropy).detach()).mean()
                    entropy_losses.append(entropy_loss.item())

                    self.entropy_optimizer.zero_grad()
                    entropy_loss.backward()
                    self.entropy_optimizer.step()

                # Compute the target values from the current policy
                with torch.no_grad():
                    # Select next action based on current policy
                    next_action, action_info = self.policy.act(mini_batch['next_state'])
                    next_log_prob = action_info['log_prob']

                    # Compute the Q-values for all critics using target networks
                    next_q_values = self.policy.get_target_q_values(obs=mini_batch['next_state'], action=next_action).squeeze(-1)
                    next_q_values = torch.min(next_q_values, dim=1, keepdim=True)[0]

                    # Add the entropy term to the target Q-values
                    next_q_values += -entropy_coeff * next_log_prob

                    # Compute the discounted target Q-values
                    y = mini_batch['reward'] + (1 - mini_batch['done'].float()) * self.config.gamma * next_q_values

                # Sum the losses across all critics
                qs = [self.policy.get_q_values(mini_batch['state'].detach(), mini_batch['action'].detach(), index=i) for i in range(self.policy.num_critics)]
                critic_loss = sum(F.mse_loss(y, q) for q in qs)
                critics_losses.append(critic_loss.item())

                # Take a gradient step on the critics
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Compute Actor loss
                q_values_pi = self.policy.get_q_values(obs=mini_batch['state'], action=current_action)
                q_values_pi = torch.min(q_values_pi, dim=1, keepdim=True)[0]
                actor_loss = (entropy_coeff * current_log_prob - q_values_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update target critic networks
                for i in range(self.policy.num_critics):
                    utils.polyak_update(self.policy.target_critics[i], self.policy.critics[i], tau=self.config.tau)   

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
                logger.log_scalar(f'critic_loss', critic_loss.item(), num_steps)

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
            "algo": "SAC",
            "agent_format_version": 1,
            "config": asdict(self.config),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        if self.entropy_optimizer is not None:
            payload["entropy_optimizer_state_dict"] = self.entropy_optimizer.state_dict()

        torch.save(payload, path / "agent.pt")
        self.policy.save(path / "policy.pt")


    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "SACAgent":
        """
        Loads the checkpoint and returns a fully-constructed SACAgent.
        """
        p = Path(path)
        agent_meta = torch.load(p / "agent.pt", map_location=map_location, weights_only=False)

        if agent_meta.get("algo") != "SAC":
            raise ValueError(f"Checkpoint algo mismatch: expected SAC, got {agent_meta.get('algo')}")

        config = SACConfig(**agent_meta["config"])
        policy = SACPolicy.load(p / "policy.pt", map_location=map_location)

        agent = cls(
            policy=policy,
            config=config,
            device=str(map_location),
        )

        actor_opt_state = agent_meta["actor_optimizer_state_dict"]
        critic_opt_state = agent_meta["critic_optimizer_state_dict"]
        agent.actor_optimizer.load_state_dict(actor_opt_state)
        agent.critic_optimizer.load_state_dict(critic_opt_state)

        if "entropy_optimizer_state_dict" in agent_meta:
            entropy_opt_state = agent_meta["entropy_optimizer_state_dict"]
            if agent.entropy_optimizer is not None:
                agent.entropy_optimizer.load_state_dict(entropy_opt_state)
            else:
                print("Warning: checkpoint has entropy optimizer state but current agent does not have an entropy optimizer. Skipping loading entropy optimizer state.")
        
        return agent