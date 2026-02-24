"""
Policy Gradient algorithm
=========================

Example Usage:
--------------
This example demonstrates how to initialize a Policy Gradient agent with a custom policy.

"""
import copy
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import itertools
import torch
from torch import nn, Tensor
from typing import List, Tuple, Dict
from prt_rl.agent import Agent
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
from prt_rl.common.collectors import Collector
from prt_rl.common.policies import NeuralPolicy
from prt_rl.common.components.heads import DistributionHead, ValueHead
import prt_rl.common.utils as utils

@dataclass
class PolicyGradientConfig:
    """
    Hyperparameter Configuration for the Policy Gradient agent.
    
    Args:
        batch_size (int): Size of the batch for training. Default is 100.
        learning_rate (float): Learning rate for the optimizer. Default is 1e-3.
        gamma (float): Discount factor for future rewards. Default is 0.99.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation. Default is 0.95.
        optim_steps (int): Number of optimization steps per training iteration. Default is 1.
        reward_to_go (bool): Whether to use rewards-to-go instead of total discounted return. Default is False.
        use_gae (bool): Whether to use Generalized Advantage Estimation. Default is False.
        baseline_learning_rate (float): Learning rate for the baseline network if used. Default is 5e-3.
        baseline_optim_steps (int): Number of optimization steps for the baseline network. Default is 5.
        normalize_advantages (bool): Whether to normalize advantages before training. Default is True.
    """
    batch_size: int = 100
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    optim_steps: int = 1
    use_reward_to_go: bool = False
    use_gae: bool = False
    baseline_learning_rate: float = 5e-3
    baseline_optim_steps: int = 5
    normalize_advantages: bool = True

class PolicyGradientPolicy(NeuralPolicy):
    """
    Base class for Policy Gradient policies. This class can be extended to create custom policies for the Policy Gradient agent.
    The policy should output a distribution over actions given the current state.

    Args:
        network (nn.Module): The neural network that processes the input state and outputs a latent representation.
        actor_head (DistributionHead): The head that takes the latent representation from the network and outputs a distribution over actions.
        critic_head (ValueHead): The head that takes the latent representation from the network and outputs a value estimate for the state.
        device (str): Device to run the policy on (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
    """
    def __init__(self,
                 network: nn.Module,
                 actor_head: DistributionHead,
                 critic_head: ValueHead | None = None,
                 *,
                 critic_network: nn.Module | None = None,
                 ) -> None:
        super().__init__()
        self.network = network
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.critic_network = critic_network

        if critic_head is not None and critic_network is None:
            self.critic_network = copy.deepcopy(network)

    # @torch.no_grad()
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

        return action, {"log_prob": log_prob}  
    
    def get_state_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns the state value for the given observation.
        
        Args:
          obs:      (B, obs_dim)
        Returns:
          value:    (B,1)
        """        
        if self.critic_head is None:
            raise ValueError("Critic head is not defined for this policy.")
        
        latent = self.critic_network(obs)
        value = self.critic_head(latent)
        return value

class PolicyGradientAgent(Agent):
    """
    Policy Gradient agent with step-wise optimization.

    Example:
        .. code-block:: python

            from prt_rl import PolicyGradient
            from prt_rl.common.policies import DistributionPolicy

            # Setup the environment
            # env = ...

            # Configure the Algorithm Hyperparameters
            config = PolicyGradientConfig(
                batch_size=1000,
                learning_rate=5e-3,
                gamma=1.0,
                use_reward_to_go=True,
                normalize_advantages=True,
            )

            # Configure Policy Gradient Policy
            policy = DistributionPolicy(env_params=env.get_parameters())

            # Create Agent
            agent = PolicyGradient(policy=policy, config=config)

            # Train the agent
            agent.train(env=env, total_steps=num_iterations * config.batch_size)    

    Args:
        config (PolicyGradientConfig): Configuration for the Policy Gradient agent.
        policy (PolicyGradientPolicy): The policy to be used by the agent.
        device (str): Device to run the agent on (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
    """
    def __init__(self, 
                 policy: PolicyGradientPolicy,
                 config: PolicyGradientConfig = PolicyGradientConfig(),
                 *,
                 device: str = 'cpu',
                 ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)

        self.policy = policy 
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(itertools.chain(self.policy.network.parameters(), self.policy.actor_head.parameters()), lr=self.config.learning_rate)
        self.critic_optimizer = None  

        if self.policy.critic_head is not None:
            self.critic_optimizer = torch.optim.Adam(itertools.chain(self.policy.critic_network.parameters(), self.policy.critic_head.parameters()), lr=self.config.baseline_learning_rate)

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        action, _ = self.policy.act(obs, deterministic=deterministic)
        return action
    
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: List[ParameterScheduler] = [],
              logger: Logger | None = None,
              evaluator: Evaluator | None = None,
              show_progress: bool = True
              ) -> None:
        """
        Train the PolicyGradient agent using the provided environment

        Args:
            env (EnvironmentInterface): The environment in which the agent will operate.
            total_steps (int): Total number of training steps to perform.
            schedulers (List[ParameterScheduler]): List of parameter schedulers to update during training.
            logger (Optional[Logger]): Logger for logging training progress. If None, a default logger will be created.
            evaluator (Evaluator): Evaluator to evaluate the agent periodically.
            show_progress (bool): If True, show a progress bar during training.
        """
        logger = logger or Logger()
        evaluator = evaluator or Evaluator()

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        # Initialize collector without flattening so the experience shape is (B, ...)
        collector = Collector(env=env, logger=logger, flatten=True)

        num_steps = 0
        while num_steps < total_steps:
            self._update_schedulers(schedulers=schedulers, step=num_steps, logger=logger)

            # Collect experience using the current policy
            trajectories = collector.collect_trajectory(policy=self.policy, min_num_steps=self.config.batch_size)
            num_steps += trajectories['state'].shape[0]  

            # Compute Monte Carlo estimate of the Q function
            if self.config.use_gae:
                values = self.policy.get_state_value(trajectories['state']).detach()
                advantages, Q_hat = utils.generalized_advantage_estimates(
                    rewards=trajectories['reward'],
                    values=values,
                    dones=trajectories['done'],
                    last_values=trajectories['last_value'] if 'last_value' in trajectories else torch.zeros_like(values[-1]),
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda
                )
            else:
                if self.config.use_reward_to_go:
                    # \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}
                    Q_hat = utils.rewards_to_go(
                        rewards=trajectories['reward'],
                        dones=trajectories['done'],
                        gamma=self.config.gamma
                    )
                else:
                    # Total discounted return               
                    # \sum_{t'=0}^{T-1} \gamma^t r_t'
                    Q_hat = utils.trajectory_returns(
                        rewards=trajectories['reward'],
                        dones=trajectories['done'],
                        gamma=self.config.gamma
                    )

                advantages = Q_hat

                # Subtract baseline if applicable
                if self.policy.critic_head is not None:
                    advantages -= self.policy.get_state_value(trajectories['state'])
            
            loss = self._compute_loss(advantages, trajectories['log_prob'], self.config.normalize_advantages)
            save_loss = loss.item()

            self.optimizer.zero_grad()
            loss.backward()  # Backpropagate the loss
            self.optimizer.step()  # Update the policy parameters
            
            # Update the baseline is applicable
            critic_losses = None
            if self.policy.critic_head is not None:
                critic_losses = []
                for _ in range(self.config.baseline_optim_steps):
                    # Compute the Q function predictions
                    q_value_pred = self.policy.get_state_value(trajectories['state'])

                    critic_loss = torch.nn.functional.mse_loss(q_value_pred, Q_hat)
                    critic_losses.append(critic_loss.item())

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()  # Backpropagate the critic loss
                    self.critic_optimizer.step()  # Update the critic parameters
                    
            if show_progress:
                tracker = collector.get_metric_tracker()
                progress_bar.update(num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}, "
                                                                   f"Episode Length: {tracker.last_episode_length}, "
                                                                   f"Loss: {save_loss:.4f},")

            # Log the training progress
            if logger.should_log(num_steps):
                logger.log_scalar("policy_loss", save_loss, iteration=num_steps)
                if critic_losses is not None:
                    logger.log_scalar("critic_loss", np.mean(critic_losses), iteration=num_steps)

            # Evaluate the agent periodically
            evaluator.evaluate(agent=self.policy, iteration=num_steps)
        
        evaluator.close()

    @classmethod
    def _compute_loss(cls, 
                      advantages, 
                      log_probs, 
                      normalize
                      ) -> torch.Tensor:
        """
        Compute the loss for the policy gradient update.

        Args:
            advantages (List[torch.Tensor]): List of advantages for each trajectory with shape (B, 1)
            log_probs (List[torch.Tensor]): List of log probabilities for each trajectory with shape (B, 1)
            normalize (bool): Whether to normalize the advantages.

        Returns:
            torch.Tensor: Computed loss value.
        """
        if normalize:
            advantages = utils.normalize_advantages(advantages)
        
        loss = -(log_probs * advantages).mean()
        return loss
    
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
            "algo": "PolicyGradient",
            "agent_format_version": 1,
            "config": asdict(self.config),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.critic_optimizer is not None:
            payload["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()

        torch.save(payload, path / "agent.pt")
        self.policy.save(path / "policy.pt")


    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PolicyGradientAgent":
        """
        Loads the checkpoint and returns a fully-constructed PolicyGradientAgent.
        """
        p = Path(path)
        agent_meta = torch.load(p / "agent.pt", map_location=map_location, weights_only=False)

        if agent_meta.get("algo") != "PolicyGradient":
            raise ValueError(f"Checkpoint algo mismatch: expected PolicyGradient, got {agent_meta.get('algo')}")

        config = PolicyGradientConfig(**agent_meta["config"])
        policy = PolicyGradientPolicy.load(p / "policy.pt", map_location=map_location)

        agent = cls(
            policy=policy,
            config=config,
            device=str(map_location),
        )

        opt_state = agent_meta["optimizer_state_dict"]
        agent.optimizer.load_state_dict(opt_state)

        if "critic_optimizer_state_dict" in agent_meta:
            agent.critic_optimizer.load_state_dict(agent_meta["critic_optimizer_state_dict"])

        return agent      