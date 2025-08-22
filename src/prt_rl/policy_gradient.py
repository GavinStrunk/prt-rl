"""
Policy Gradient algorithm
=========================

Example Usage:
--------------
This example demonstrates how to initialize a Policy Gradient agent with a custom policy.

.. code-block:: python

    from prt_rl import PolicyGradient
    from prt_rl.common.policies import DistributionPolicy
    from prt_rl.common.networks import MLP
    from prt_rl.common.distributions import Normal

    env = ...  # Initialize your environment here
    agent = PolicyGradient(
        env_params=env.get_env_params(),
        policy=DistributionPolicy(
            env_params=env.get_env_params(),
            encoder_network=None,
            encoder_kwargs={},
            policy_head=MLP,
            policy_kwargs={'network_arch': [64, 64]},
            distribution=Normal
            ),
        learning_rate=1e-3,
        ...
        )

    agent.train(env, total_steps=10000)
"""
from dataclasses import dataclass
import numpy as np
import torch
from typing import List
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvironmentInterface, EnvParams
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
from prt_rl.common.collectors import SequentialCollector, ParallelCollector
from prt_rl.common.policies import DistributionPolicy
from prt_rl.common.networks import MLP
import prt_rl.common.utils as utils

@dataclass
class PolicyGradientConfig:
    """
    Hyperparameters for the Policy Gradient agent.
    
    Args:
        batch_size (int): Size of the batch for training. Default is 100.
        learning_rate (float): Learning rate for the optimizer. Default is 1e-3.
        gamma (float): Discount factor for future rewards. Default is 0.99.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation. Default is 0.95.
        optim_steps (int): Number of optimization steps per training iteration. Default is 1.
        reward_to_go (bool): Whether to use rewards-to-go instead of total discounted return. Default is False.
        use_baseline (bool): Whether to use a baseline for advantage estimation. Default is False.
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
    use_baseline: bool = False
    use_gae: bool = False
    baseline_learning_rate: float = 5e-3
    baseline_optim_steps: int = 5
    normalize_advantages: bool = True

class PolicyGradient(BaseAgent):
    """
    Policy Gradient agent with step-wise optimization.

    Args:
        env_params (EnvParams): Environment parameters.
        config (PolicyGradientConfig): Configuration for the Policy Gradient agent.
        policy (Optional[DistributionPolicy]): The policy to be used by the agent. If None, a default policy will be created based on the environment parameters.
        device (str): Device to run the agent on (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
    """
    def __init__(self, 
                 env_params: EnvParams,
                 config: PolicyGradientConfig = PolicyGradientConfig(),
                 policy: DistributionPolicy | None = None,
                 device: str = 'cpu',
                 ) -> None:
        super().__init__()
        self.env_params = env_params
        self.config = config
        self.device = torch.device(device)

        self.policy = policy if policy is not None else DistributionPolicy(env_params=env_params)
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

        if self.config.use_baseline or self.config.use_gae:
            self.critic = MLP(
                input_dim=env_params.observation_shape[0],
                output_dim=1,
                network_arch=[64,64],
            )
            self.critic.to(self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.baseline_learning_rate)
        else:
            self.critic = None


    def predict(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Perform an action based on the current state using the policy.

        Args:
            state (torch.Tensor): Current state of the environment.
            deterministic (bool): If True, use the deterministic action from the policy. Default is False.

        Returns:
            torch.Tensor: Action to be taken by the agent.
        """
        with torch.no_grad():
            return self.policy(state, deterministic=deterministic)  # Forward pass through the policy
    
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
        logger = logger or Logger.create('blank')

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        # Initialize collector without flattening so the experience shape is (B, ...)
        collector = ParallelCollector(env=env, logger=logger, flatten=True)

        num_steps = 0
        while num_steps < total_steps:
            # Update schedulers if any
            for scheduler in schedulers:
                scheduler.update(current_step=num_steps)

            # Collect experience using the current policy
            trajectories = collector.collect_trajectory(policy=self.policy, min_num_steps=self.config.batch_size)
            num_steps += trajectories['state'].shape[0]  

            # Compute Monte Carlo estimate of the Q function
            if self.config.use_gae:
                values = self.critic(trajectories['state']).detach()
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

                if self.config.use_baseline:
                    advantages = Q_hat - self.critic(trajectories['state']).squeeze(1)
                else:
                    advantages = Q_hat
            
            loss = self._compute_loss(advantages, trajectories['log_prob'], self.config.normalize_advantages)
            save_loss = loss.item()

            self.optimizer.zero_grad()
            loss.backward()  # Backpropagate the loss
            self.optimizer.step()  # Update the policy parameters
            
            # Update the baseline is applicable
            if self.critic is not None:
                critic_losses = []
                for _ in range(self.config.baseline_optim_steps):
                    # Compute the Q function predictions
                    q_value_pred = self.critic(trajectories['state']).squeeze(1)

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
                if self.critic is not None:
                    logger.log_scalar("critic_loss", np.mean(critic_losses), iteration=num_steps)

            # Evaluate the agent periodically
            if evaluator is not None:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)
        
        if evaluator is not None:
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

    # @staticmethod        
    # def _compute_trajectory_rewards(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    #     """
    #     Compute the total discounted return G from a full trajectory.

    #     ..math::
    #         \hat{Q}(s_{i,t},a_{i,t}) = \sum_{t'=1}^{T} \gamma^t r(s_{i,t}, a_{i,t})
        
    #     Args:
    #         rewards: Tensor of shape (B,) with rewards for each timestep.
    #         dones: Tensor of shape (B,) with done flags indicating if the episode has ended.
    #         gamma: Discount factor

    #     Returns:
    #         Scalar float representing total discounted return with shape (B, )
    #     """
    #     returns = []
    #     start = 0

    #     for t in range(len(rewards)):
    #         if dones[t]:
    #             # Slice out the trajectory
    #             trajectory = rewards[start:t+1]
    #             T = trajectory.shape[0]
    #             total_return = trajectory.sum() * (gamma ** T)
    #             # Fill trajectory with the same return value
    #             filled = torch.full((T,), total_return.item(), dtype=torch.float32, device=rewards.device)
    #             returns.append(filled)
    #             start = t + 1

    #     return torch.cat(returns, dim=0)
    
    # @staticmethod
    # def _compute_rewards_to_go(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    #     """
    #     Compute rewards-to-go from rewards and done flags.

    #     Args:
    #         rewards (torch.Tensor): Rewards from the environment with shape (B, 1).
    #         dones (torch.Tensor): Done flags indicating if the episode has ended with shape (N, T, 1).
    #         gamma (float): Discount factor.

    #     Returns:
    #         torch.Tensor: Computed rewards-to-go with shape (B, )
    #     """
    #     rewards_to_go = []
    #     R = 0
    #     for reward, done in zip(reversed(rewards), reversed(dones)):
    #         if done:
    #             R = 0.0
    #         R = reward + gamma * R
    #         rewards_to_go.insert(0, R)

    #     return torch.stack(rewards_to_go)  # Shape (B, )
    
    # @staticmethod
    # def _compute_generalized_advantage_estimate(
    #     rewards: torch.Tensor,  
    #     values: torch.Tensor,            
    #     dones: torch.Tensor,             
    #     gamma: float = 0.99,
    #     gae_lambda: float = 0.95,
    # ):
    #     """
    #     Compute GAE and TD(lambda) returns for a batched rollout.

    #     Args:
    #         rewards (B,): Rewards from rollout
    #         values (B,): Estimated state values
    #         dones (B,): Done flags (1 if episode ended at step t, else 0)
    #         gamma (float): Discount factor
    #         gae_lambda (float): GAE lambda

    #     Returns:
    #         advantages (T, N): Estimated advantage values
    #         returns (T, N): TD(lambda) returns
    #     """
    #     B = rewards.shape[0]

    #     advantages = torch.zeros((B), dtype=values.dtype, device=values.device)

    #     for t in reversed(range(B)):
    #         if dones[t]:
    #             next_value = 0.0
    #             next_advantage = 0.0
    #         else:
    #             next_value = values[t + 1]
    #             next_advantage = advantages[t+1]
            
    #         delta = rewards[t] + gamma * next_value - values[t]
    #         advantages[t] = delta + gamma * gae_lambda * next_advantage

    #     return advantages
    
    
class PolicyGradientTrajectory(PolicyGradient):
    """
    Policy Gradient agent with trajectory-based training.
    
    This class extends the PolicyGradient class to handle trajectory-based training.
    It collects trajectories and computes advantages based on the collected data.

    Args:
        env_params (EnvParams): Environment parameters.
        policy (Optional[DistributionPolicy]): The policy to be used by the agent. If None, a default policy will be created based on the environment parameters.
        batch_size (int): Size of the batch for training. Default is 100.
        learning_rate (float): Learning rate for the optimizer. Default is 1e-3.
        gamma (float): Discount factor for future rewards. Default is 0.99.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation. Default is 0.95.
        optim_steps (int): Number of optimization steps per training iteration. Default is 1.
        reward_to_go (bool): Whether to use rewards-to-go instead of total discounted return. Default is False.
        use_baseline (bool): Whether to use a baseline for advantage estimation. Default is False.
        use_gae (bool): Whether to use Generalized Advantage Estimation. Default is False.
        baseline_learning_rate (float): Learning rate for the baseline network if used. Default is 5e-3.
        baseline_optim_steps (int): Number of optimization steps for the baseline network. Default is 5.
        normalize_advantages (bool): Whether to normalize advantages before training. Default is True.
        network_arch (List[int]): Architecture of the neural network for the policy. Default is [64, 64].
        device (str): Device to run the agent on (e.g., 'cpu' or 'cuda'). Default is 'cpu'.    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        logger = logger or Logger.create('blank')

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        # Initialize collector without flattening so the experience shape is (N, T, ...)
        collector = SequentialCollector(env=env, logger=logger)

        num_steps = 0
        while num_steps < total_steps:
            # Update schedulers if any
            for scheduler in schedulers:
                scheduler.update(current_step=num_steps)

            # Collect experience using the current policy
            # trajectories is a list of tensors, each tensor is a trajectory of shape (T_i, ...)
            trajectories, batch_steps = collector.collect_trajectory(policy=self.policy, min_num_steps=self.batch_size)
            num_steps += batch_steps

            batch_states = torch.cat([t['state'] for t in trajectories], dim=0)  # Shape (N*T, D)

            # Compute Monte Carlo estimate of the Q function
            if not self.use_reward_to_go:
                # \sum_{t'=0}^{T-1} \gamma^t r_t'
                Q_hat = self._compute_trajectory_rewards(
                    rewards=[t['reward'] for t in trajectories],
                    gamma=self.gamma
                )
            else:
                # \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}
                Q_hat = self._compute_rewards_to_go(
                    rewards=[t['reward'] for t in trajectories],
                    dones=[t['done'] for t in trajectories],
                    gamma=self.gamma
                )

            if self.use_baseline:
                advantages = Q_hat - self.critic(batch_states)
            else:
                advantages = Q_hat

            loss = self._compute_loss(advantages, [t['log_prob'] for t in trajectories], self.normalize_advantages)

            self.optimizer.zero_grad()
            loss.backward()  # Backpropagate the loss
            self.optimizer.step()  # Update the policy parameters
            
            # Update the baseline is applicable
            if self.use_baseline:
                Q_hat = torch.cat(Q_hat, dim=0)  # Shape (N*T, 1)
                for _ in range(self.baseline_optim_steps):
                    batch_state = batch_states
                    q_value_pred = self.critic(batch_state).squeeze()
                    critic_loss = torch.nn.functional.mse_loss(q_value_pred, Q_hat)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()  # Backpropagate the critic loss
                    self.critic_optimizer.step()  # Update the critic parameters
                    
            if show_progress:
                progress_bar.update(num_steps, desc=f"Episode Reward: {collector.previous_episode_reward:.2f}, "
                                                                   f"Episode Length: {collector.previous_episode_length}, "
                                                                   f"Loss: {loss:.4f},")

            # Log the training progress
            if logger.should_log(num_steps):
                pass

            # Evaluate the agent periodically
            if evaluator is not None:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)
        
        if evaluator is not None:
            evaluator.close()
            
    def _compute_loss(self, advantages, log_probs, normalize):

        if normalize:
            # Compute the mean and std of the advantages across all trajectories otherwise the advantages will all be 0 for the total discounted return
            flat_adv = torch.cat(advantages)
            mean_adv = flat_adv.mean()
            std_adv = flat_adv.std()
            for i in range(len(advantages)):
                advantages[i] = (advantages[i] - mean_adv) / (std_adv + 1e-8)
        # Update the policy using the computed advantages
        # Trajectories is [T_i, ...]_N and advantages is [T_i, 1]_N
        losses = []
        for log_prob, adv in zip(log_probs, advantages):
            log_prob_sum = log_prob * adv.unsqueeze(1)  # Shape (T_i, 1)
            losses.append(log_prob_sum.sum())
        losses = torch.stack(losses, dim=0)
        loss = -losses.mean()        
        return loss    
    
    @staticmethod        
    def _compute_trajectory_rewards(rewards: List[torch.Tensor], gamma: float):
        """
        Compute the total discounted return G from a full trajectory.

        ..math::
            \hat{Q}
        
        Args:
            rewards: Tensor of shape (N, T, 1) with rewards for each timestep.
            gamma: Discount factor

        Returns:
            Scalar float representing total discounted return with shape (N, T)
        """
        returns = []
        for r in rewards:
            T = r.shape[0]
            discounts = gamma ** torch.arange(T, dtype=torch.float32, device=r.device).unsqueeze(1)
            total_return = torch.sum(discounts * r).item()
            returns.append(torch.full((T,), total_return, dtype=torch.float32, device=r.device))
        return returns
    
    @staticmethod
    def _compute_rewards_to_go(rewards: List[torch.Tensor], dones: List[torch.Tensor], gamma: float) -> List[torch.Tensor]:
        """
        Compute rewards-to-go from rewards and done flags.

        Args:
            rewards (torch.Tensor): Rewards from the environment with shape (N, T, 1).
            dones (torch.Tensor): Done flags indicating if the episode has ended with shape (N, T, 1).
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Computed rewards-to-go with shape (N, T)
        """
        rewards_to_go = []
        for reward_traj, done_traj in zip(rewards, dones):
            returns = []
            R = 0.0
            for reward, done in zip(reversed(reward_traj), reversed(done_traj)):
                if done:
                    R = 0.0
                R = reward + gamma * R
                returns.insert(0, R)
            rewards_to_go.append(torch.tensor(returns, dtype=torch.float32, device=reward_traj.device))

        return rewards_to_go 