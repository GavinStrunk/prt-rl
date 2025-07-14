import numpy as np
import torch
from typing import Optional, List
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvironmentInterface, EnvParams
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator

from prt_rl.common.collectors import SequentialCollector
from prt_rl.common.policies import DistributionPolicy
from prt_rl.common.networks import MLP


class PolicyGradient(BaseAgent):
    """
    Policy Gradient agent with step-wise optimization.

    """
    def __init__(self, 
                 env_params: EnvParams,
                 policy = None,
                 batch_size: int = 100,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 optim_steps: int = 1,
                 reward_to_go: bool = False, 
                 use_baseline: bool = False,
                 use_gae: bool = False,
                 baseline_learning_rate: float = 5e-3,
                 baseline_optim_steps: int = 5,
                 normalize_advantages: bool = True,
                 network_arch: List[int] = [64, 64],
                 device: str = 'cpu',
                 ) -> None:
        # Construct the policy if not provided
        policy = policy if policy is not None else DistributionPolicy(env_params=env_params, policy_kwargs={'network_arch': network_arch}, device=device)
        super().__init__(policy=policy)

        self.env_params = env_params
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_to_go = reward_to_go
        self.use_baseline = use_baseline
        self.use_gae = use_gae
        self.baseline_learning_rate = baseline_learning_rate
        self.baseline_optim_steps = baseline_optim_steps
        self.normalize_advantages = normalize_advantages
        self.optim_steps = optim_steps
        self.network_arch = network_arch
        self.device = torch.device(device)

        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        if use_baseline:
            self.critic = MLP(
                input_dim=env_params.observation_shape[0],
                output_dim=1,
                network_arch=self.network_arch,
            )
            self.critic.to(self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=baseline_learning_rate)
        else:
            self.critic = None


    def predict(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy(state)  # Forward pass through the policy
    
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: List[ParameterScheduler] = [],
              logger: Optional[Logger] = None,
              logging_freq: int = 1,
              evaluator: Evaluator = Evaluator(),
              eval_freq: int = 1000,
              ) -> None:
        """
        Train the PolicyGradient agent using the provided environment

        Args:
            env (EnvironmentInterface): The environment in which the agent will operate.
            total_steps (int): Total number of training steps to perform.
            schedulers (List[ParameterScheduler]): List of parameter schedulers to update during training.
            logger (Optional[Logger]): Logger for logging training progress. If None, a default logger will be created.
            logging_freq (int): Frequency of logging training progress.
            evaluator (Evaluator): Evaluator to evaluate the agent periodically.
            eval_freq (int): Frequency of evaluation during training.
        """
        logger = logger or Logger.create('blank')
        progress_bar = ProgressBar(total_steps=total_steps)

        # Initialize collector without flattening so the experience shape is (N, T, ...)
        collector = SequentialCollector(env=env, logger=logger, logging_freq=logging_freq)

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
            batch_rewards = torch.cat([t['reward'] for t in trajectories], dim=0).squeeze(1)  # Shape (N*T, 1)
            batch_dones = torch.cat([t['done'] for t in trajectories], dim=0).squeeze(1)  # Shape (N*T, 1)
            batch_log_probs = torch.cat([t['log_prob'] for t in trajectories], dim=0).squeeze(1)  # Shape (N*T,)

            # Compute Monte Carlo estimate of the Q function
            if self.use_gae:
                values = self.critic(batch_states).detach().squeeze(1)
                advantages = self._compute_generalized_advantage_estimate(
                    rewards=batch_rewards,
                    dones=batch_dones,
                    values=values,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda
                )
                Q_hat = advantages + values
            elif not self.reward_to_go:
                # \sum_{t'=0}^{T-1} \gamma^t r_t'
                Q_hat = self._compute_trajectory_rewards(
                    rewards=batch_rewards,
                    dones=batch_dones,
                    gamma=self.gamma
                )
                advantages = Q_hat - self.critic(batch_states).squeeze(1) if self.critic is not None else Q_hat
            else:
                # \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}
                Q_hat = self._compute_rewards_to_go(
                    rewards=batch_rewards,
                    dones=batch_dones,
                    gamma=self.gamma
                )
                advantages = Q_hat - self.critic(batch_states).squeeze(1) if self.critic is not None else Q_hat

            loss = self._compute_loss(advantages, batch_log_probs, self.normalize_advantages)

            self.optimizer.zero_grad()
            loss.backward()  # Backpropagate the loss
            self.optimizer.step()  # Update the policy parameters
            
            # Update the baseline is applicable
            if self.critic is not None:
                for _ in range(self.baseline_optim_steps):
                    # Compute the Q function predictions
                    q_value_pred = self.critic(batch_states).squeeze(1)

                    critic_loss = torch.nn.functional.mse_loss(q_value_pred, Q_hat)

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()  # Backpropagate the critic loss
                    self.critic_optimizer.step()  # Update the critic parameters
                    
            progress_bar.update(num_steps, desc=f"Episode Reward: {collector.previous_episode_reward:.2f}, "
                                                                   f"Episode Length: {collector.previous_episode_length}, "
                                                                   f"Loss: {loss:.4f},")

            # Log the training progress
            if num_steps % logging_freq == 0:
                pass

            # Evaluate the agent periodically
            if num_steps % eval_freq == 0:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)
        
        evaluator.close()

    def _compute_loss(self, advantages, log_probs, normalize):
        """
        Compute the loss for the policy gradient update.

        Args:
            advantages (List[torch.Tensor]): List of advantages for each trajectory with shape (B, 1)
            log_probs (List[torch.Tensor]): List of log probabilities for each trajectory with shape (B, ...)
            normalize (bool): Whether to normalize the advantages.

        Returns:
            torch.Tensor: Computed loss value.
        """
        if normalize:
            advantages = self._normalize_advantages(advantages)
        
        loss = -(log_probs * advantages).mean()
        return loss

    @staticmethod        
    def _compute_trajectory_rewards(rewards: torch.Tensor, dones: torch.Tensor, gamma: float):
        """
        Compute the total discounted return G from a full trajectory.

        ..math::
            \hat{Q}(s_{i,t},a_{i,t}) = \sum_{t'=1}^{T} \gamma^t r(s_{i,t}, a_{i,t})
        
        Args:
            rewards: Tensor of shape (B,) with rewards for each timestep.
            dones: Tensor of shape (B,) with done flags indicating if the episode has ended.
            gamma: Discount factor

        Returns:
            Scalar float representing total discounted return with shape (B, )
        """
        returns = []
        start = 0

        for t in range(len(rewards)):
            if dones[t]:
                # Slice out the trajectory
                trajectory = rewards[start:t+1]
                T = trajectory.shape[0]
                total_return = trajectory.sum() * (gamma ** T)
                # Fill trajectory with the same return value
                filled = torch.full((T,), total_return.item(), dtype=torch.float32, device=rewards.device)
                returns.append(filled)
                start = t + 1

        return torch.cat(returns, dim=0)
    
    @staticmethod
    def _compute_rewards_to_go(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Compute rewards-to-go from rewards and done flags.

        Args:
            rewards (torch.Tensor): Rewards from the environment with shape (B, 1).
            dones (torch.Tensor): Done flags indicating if the episode has ended with shape (N, T, 1).
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Computed rewards-to-go with shape (B, )
        """
        rewards_to_go = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0.0
            R = reward + gamma * R
            rewards_to_go.insert(0, R)

        return torch.stack(rewards_to_go)  # Shape (B, )
    
    @staticmethod
    def _compute_generalized_advantage_estimate(
        rewards: torch.Tensor,  
        values: torch.Tensor,            
        dones: torch.Tensor,             
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE and TD(lambda) returns for a batched rollout.

        Args:
            rewards (B,): Rewards from rollout
            values (B,): Estimated state values
            dones (B,): Done flags (1 if episode ended at step t, else 0)
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda

        Returns:
            advantages (T, N): Estimated advantage values
            returns (T, N): TD(lambda) returns
        """
        B = rewards.shape[0]

        advantages = torch.zeros((B), dtype=values.dtype, device=values.device)

        for t in reversed(range(B)):
            if dones[t]:
                next_value = 0.0
                next_advantage = 0.0
            else:
                next_value = values[t + 1]
                next_advantage = advantages[t+1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * gae_lambda * next_advantage

        return advantages
    
    @staticmethod
    def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
        """
        Normalize advantages to have zero mean and unit variance.

        Args:
            advantages (torch.Tensor): Advantages to normalize with shape (B,).

        Returns:
            torch.Tensor: Normalized advantages with shape (B,).
        """
        mean = advantages.mean()
        std = advantages.std()
        normalized_advantages = (advantages - mean) / (std + 1e-8)
        return normalized_advantages
    
class PolicyGradientTrajectory(PolicyGradient):
    """
    Policy Gradient agent with trajectory-based training.
    
    This class extends the PolicyGradient class to handle trajectory-based training.
    It collects trajectories and computes advantages based on the collected data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: List[ParameterScheduler] = [],
              logger: Optional[Logger] = None,
              logging_freq: int = 1,
              evaluator: Evaluator = Evaluator(),
              eval_freq: int = 1000,
              ) -> None:
        """
        Train the PolicyGradient agent using the provided environment

        Args:
            env (EnvironmentInterface): The environment in which the agent will operate.
            total_steps (int): Total number of training steps to perform.
            schedulers (List[ParameterScheduler]): List of parameter schedulers to update during training.
            logger (Optional[Logger]): Logger for logging training progress. If None, a default logger will be created.
            logging_freq (int): Frequency of logging training progress.
            evaluator (Evaluator): Evaluator to evaluate the agent periodically.
            eval_freq (int): Frequency of evaluation during training.
        """
        logger = logger or Logger.create('blank')
        progress_bar = ProgressBar(total_steps=total_steps)

        # Initialize collector without flattening so the experience shape is (N, T, ...)
        collector = SequentialCollector(env=env, logger=logger, logging_freq=logging_freq)

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
            batch_rewards = [t['reward'] for t in trajectories]  # Shape (N, T, 1)
            batch_dones = [t['done'] for t in trajectories]  # Shape (N, T, 1)
            batch_log_probs = torch.cat([t['log_prob'] for t in trajectories], dim=0)  # Shape (N*T, 1)

            # Compute Monte Carlo estimate of the Q function
            if not self.reward_to_go:
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

            # if not self.use_step_avg:
            #     if self.normalize_advantages:
            #         # Compute the mean and std of the advantages across all trajectories otherwise the advantages will all be 0 for the total discounted return
            #         flat_adv = torch.cat(advantages)
            #         mean_adv = flat_adv.mean()
            #         std_adv = flat_adv.std()
            #         for i in range(len(advantages)):
            #             advantages[i] = (advantages[i] - mean_adv) / (std_adv + 1e-8)

            #     # Update the policy using the computed advantages
            #     # Trajectories is [T_i, ...]_N and advantages is [T_i, 1]_N
            #     losses = []
            #     for traj, adv in zip(trajectories, advantages):
            #         log_prob_sum = traj['log_prob'] * adv.unsqueeze(1)  # Shape (T_i, 1)
            #         losses.append(log_prob_sum.sum())
            #     losses = torch.stack(losses, dim=0)

            #     loss = -losses.mean()
            # else:
            #     advantages = torch.cat(advantages, dim=0)  # Shape (N*T,)

            #     if self.normalize_advantages:
            #         advantages = self._normalize_advantages(advantages)
                
            #     log_probs = torch.cat([t['log_prob'].squeeze() for t in trajectories], dim=0)  # Shape (N*T, 1)
            #     loss = -(log_probs * advantages).mean()

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
                    
            progress_bar.update(num_steps, desc=f"Episode Reward: {collector.previous_episode_reward:.2f}, "
                                                                   f"Episode Length: {collector.previous_episode_length}, "
                                                                   f"Loss: {loss:.4f},")

            # Log the training progress
            if num_steps % logging_freq == 0:
                pass

            # Evaluate the agent periodically
            if num_steps % eval_freq == 0:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)
        
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