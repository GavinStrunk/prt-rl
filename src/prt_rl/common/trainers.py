import torch
from abc import ABC, abstractmethod
import copy
from tensordict.tensordict import TensorDict
from typing import Optional, List, Any
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.policy import Policy, QNetworkPolicy, ActorCriticPolicy
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.metrics import MetricTracker


class Trainer(ABC):
    def __init__(self,
                 env: EnvironmentInterface,
                 policy: Policy,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 schedulers: Optional[List[ParameterScheduler]] = None,
                 progress_bar: Optional[ProgressBar] = ProgressBar,
                 ) -> None:
        self.env = env
        self.policy = policy
        self.logger = logger or Logger()
        self.metric_tracker = metric_tracker or MetricTracker()
        self.schedulers = schedulers or []
        self.progress_bar = progress_bar

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        """
        Sets a name value parameter

        Args:
            name (str): The name of the parameter
            value (Any): The value of the parameter

        Raises:
            ValueError: If the parameter is not found
        """
        try:
            self.policy.set_parameter(name, value)
        except ValueError:
            raise ValueError(f"Parameter {name} not found in {self.__class__.__name__}")

    def get_policy(self) -> Policy:
        """
        Returns the current policy.

        Returns:
            Policy: current policy object.
        """
        return self.policy

    def save_policy(self):
        """
        Saves the current policy.
        """
        self.logger.save_policy(self.policy)

    @abstractmethod
    def train(self,
              num_episodes: int,
              num_agents: int = 1,
              ) -> None:
        raise NotImplementedError


class TDTrainer(Trainer):
    """
    Temporal Difference Reinforcement Learning (TD) trainer base class. RL algorithms are implementations of this class.

    """

    def __init__(self,
                 env: EnvironmentInterface,
                 policy: Policy,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 schedulers: Optional[List[ParameterScheduler]] = None,
                 progress_bar: Optional[ProgressBar] = ProgressBar,
                 ) -> None:
        super().__init__(env=env, policy=policy, logger=logger, metric_tracker=metric_tracker, schedulers=schedulers,
                         progress_bar=progress_bar)

    @abstractmethod
    def update_policy(self,
                      experience: TensorDict,
                      ) -> None:
        raise NotImplementedError

    def train(self,
              num_episodes: int,
              num_agents: int = 1,
              ) -> None:
        # Initialize progress bar
        if self.progress_bar is not None:
            self.progress_bar = self.progress_bar(total_frames=num_episodes, frames_per_batch=1)

        # Create agent copies
        agents = []
        for _ in range(num_agents):
            agents.append(copy.deepcopy(self.policy))

        cumulative_reward = 0
        # Initialize metrics
        for i in range(num_episodes):

            # Step schedulers if there are any
            for sch in self.schedulers:
                name = sch.parameter_name
                new_val = sch.update(i)
                self.set_parameter(name, new_val)
                self.logger.log_scalar(name, new_val, iteration=i)

            obs_td = self.env.reset()
            done = False

            # Pre-episode metrics
            episode_reward = 0
            while not done:
                action_td = self.policy.get_action(obs_td)
                # Save action choice
                obs_td = self.env.step(action_td)
                self.update_policy(obs_td)
                episode_reward += obs_td['next', 'reward']
                done = obs_td['next', 'done']

                # Compute post step metrics
                obs_td = self.env.step_mdp(obs_td)

            # Compute post episode metrics
            cumulative_reward += episode_reward
            self.progress_bar.update(episode_reward, cumulative_reward)
            self.logger.log_scalar('episode_reward', episode_reward, iteration=i)
            self.logger.log_scalar('cumulative_reward', cumulative_reward, iteration=i)


class ANNTrainer(TDTrainer):
    def __init__(self,
                 env: EnvironmentInterface,
                 policy: QNetworkPolicy,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 schedulers: Optional[List[ParameterScheduler]] = None,
                 progress_bar: Optional[ProgressBar] = ProgressBar,
                 ) -> None:
        super().__init__(env, policy=policy, logger=logger, metric_tracker=metric_tracker, schedulers=schedulers,
                         progress_bar=progress_bar)

    def get_policy_network(self):
        return self.policy.q_network

    @abstractmethod
    def update_policy(self,
                      experience: TensorDict,
                      ) -> None:
        raise NotImplementedError


class ActorCriticTrainer(Trainer):
    def __init__(self,
                 env: EnvironmentInterface,
                 policy: ActorCriticPolicy,
                 num_optimization_steps: int,
                 mini_batch_size: int,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 schedulers: Optional[List[ParameterScheduler]] = None,
                 progress_bar: Optional[ProgressBar] = ProgressBar,
                 ) -> None:
        super().__init__(env=env, policy=policy, logger=logger, metric_tracker=metric_tracker, schedulers=schedulers,
                         progress_bar=progress_bar)
        self.num_optimization_steps = num_optimization_steps
        self.mini_batch_size = mini_batch_size
        self.optimizers = None

    @abstractmethod
    def configure_optimizers(self) -> List:
        raise NotImplementedError

    def collect_experience(self):
        experience_buffer = []
        value_estimates = []
        action_log_probs = []
        obs_td = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            action_td = self.policy.get_action(obs_td)
            value_estimates.append(self.policy.get_value_estimates())
            action_log_probs.append(self.policy.get_log_probs(action_td['action']))

            obs_td = self.env.step(action_td)

            experience_buffer.append(obs_td.copy())

            episode_reward += obs_td['next', 'reward']
            done = obs_td['next', 'done']

            obs_td = self.env.step_mdp(obs_td)

        # Stack experiences
        experience_buffer = torch.cat(experience_buffer, dim=0)
        action_log_probs = torch.cat(action_log_probs, dim=0).detach()
        value_estimates = torch.cat(value_estimates, dim=0)

        return experience_buffer, action_log_probs, value_estimates

    @abstractmethod
    def compute_returns(self,
                        rewards: torch.Tensor,
                        dones: torch.Tensor
                        ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_advantages(self,
                           returns: torch.Tensor,
                           values: torch.Tensor
                           ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self,
                     batch_experience: TensorDict,
                     batch_returns: torch.Tensor,
                     batch_advantages: torch.Tensor,
                     batch_action_log_probs: torch.Tensor,
                     ) -> List:
        pass

    def train(self,
              num_episodes: int,
              num_agents: int = 1,
              ) -> None:

        # Configure the optimizers
        self.optimizers = self.configure_optimizers()

        # Initialize progress bar
        if self.progress_bar is not None:
            self.progress_bar = self.progress_bar(total_frames=num_episodes, frames_per_batch=1)

        cumulative_reward = 0
        # Initialize metrics
        for i in range(num_episodes):

            # Step schedulers if there are any
            for sch in self.schedulers:
                name = sch.parameter_name
                new_val = sch.update(i)
                self.set_parameter(name, new_val)
                self.logger.log_scalar(name, new_val, iteration=i)

            # Collect experience by running the current policy in the environment
            experience_buffer, action_log_probs, value_estimates = self.collect_experience()

            # Compute the returns - Rewards to go
            returns = self.compute_returns(rewards=experience_buffer['next', 'reward'],
                                           dones=experience_buffer['next', 'done'])
            returns = returns.detach()

            # Compute Advantages
            advantages = self.compute_advantages(returns, value_estimates)
            advantages = advantages.detach()

            # Learning Loop
            for _ in range(self.num_optimization_steps):
                for i in range(0, len(experience_buffer), self.mini_batch_size):
                    # Get batch data
                    batch_experience = experience_buffer[i:i + self.mini_batch_size]
                    batch_returns = returns[i:i + self.mini_batch_size]
                    batch_advantages = advantages[i:i + self.mini_batch_size]
                    batch_log_probs = action_log_probs[i:i + self.mini_batch_size]

                    for opt in self.optimizers:
                        opt.zero_grad()

                    losses = self.compute_loss(
                        batch_experience=batch_experience,
                        batch_returns=batch_returns,
                        batch_advantages=batch_advantages,
                        batch_action_log_probs=batch_log_probs
                    )
                    for loss in losses:
                        loss.backward()

                    for opt in self.optimizers:
                        opt.step()

            # Episode Logging
            episode_reward = experience_buffer['next', 'reward'].sum().item()
            cumulative_reward += episode_reward
            self.progress_bar.update(episode_reward, cumulative_reward)
            self.logger.log_scalar('episode_reward', episode_reward, iteration=i)
            self.logger.log_scalar('cumulative_reward', cumulative_reward, iteration=i)
