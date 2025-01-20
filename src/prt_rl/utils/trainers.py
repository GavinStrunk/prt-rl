import torch
from abc import ABC, abstractmethod
import copy
from tensordict.tensordict import TensorDict
from typing import Optional, List, Any
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.utils.policy import Policy, QNetworkPolicy
from prt_rl.utils.loggers import Logger
from prt_rl.utils.schedulers import ParameterScheduler
from prt_rl.utils.progress_bar import ProgressBar
from prt_rl.utils.metrics import MetricTracker


class TDTrainer(ABC):
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
        self.env = env
        self.policy = policy
        self.logger = logger or Logger()
        self.metric_tracker = metric_tracker or MetricTracker()
        self.schedulers = schedulers or []
        self.progress_bar = progress_bar

    @abstractmethod
    def update_policy(self,
                      experience: TensorDict,
                      ) -> None:
        raise NotImplementedError

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
            raise ValueError(f"Parameter {name} not found in TDTrainer")

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
                episode_reward += obs_td['next','reward']
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
        super().__init__(env, policy=policy, logger=logger, metric_tracker=metric_tracker, schedulers=schedulers, progress_bar=progress_bar)

    def get_policy_network(self):
        return self.policy.q_network

    @abstractmethod
    def update_policy(self,
                      experience: TensorDict,
                      ) -> None:
        raise NotImplementedError