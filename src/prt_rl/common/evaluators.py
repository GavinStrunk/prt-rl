from abc import ABC
import copy
import math
from typing import Optional
import numpy as np
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.collectors import ParallelCollector

class Evaluator(ABC):
    """
    Base class for all evaluators in the PRT-RL framework.
    This class provides a common interface for evaluating agents in different environments with different objectives.

    Args:
        eval_freq (int): Frequency of evaluation in terms of steps, iterations, or optimization steps.
    """
    def __init__(self,
                 eval_freq: int = 1,
                ) -> None:
        """
        Initialize the evaluator with the evaluation frequency.

        Args:
            eval_freq (int): Frequency of evaluation in terms of steps, iterations, or optimization steps.
        """
        self.eval_freq = eval_freq
        self.last_evaluation_iteration = 0

    def evaluate(self, agent, iteration: int, is_last: bool = False) -> None:
        """
        Evaluate the agent's performance in the given environment.

        Args:
            agent: The agent to be evaluated.
            iteration (int): The current iteration number.
            is_last (bool): Whether this is the last evaluation.

        Returns:
            None
        """
        pass
    
    def close(self) -> None:
        """
        Close the evaluator and release any resources.
        This method can be overridden by subclasses if needed.
        """
        pass

    def _should_evaluate(self, iteration: int) -> bool:
        """
        Determine if the evaluation should be performed based on the iteration number.

        Returns True if:
        - The current iteration is a multiple of eval_freq, or
        - The current iteration is the last one and it was not evaluated due to non-divisibility.

        Args:
            iteration (int): The current iteration number.

        Returns:
            bool: True if evaluation should be performed, False otherwise.
        """
        iteration = iteration + 1  # Adjust for 0-based indexing

        current_interval = iteration // self.eval_freq
        last_interval = self.last_evaluation_iteration // self.eval_freq
        if current_interval > last_interval:
            self.last_evaluation_iteration = iteration
            return True

        return False    

class RewardEvaluator(Evaluator):
    """
    Evaluators are used to assess the performance of agents or policies.

    It is important that the eval_freq value is the same units as the iteration value passed to the evaluate method. For example, if the eval_freq is set in steps then num_steps should be used as the iteration value. This ensures the evaluations occur at the correct time.

    Args:
        env (EnvironmentInterface): The environment to evaluate the agent in.
        num_episodes (int): The number of episodes to run for evaluation.
        logger (Optional[Logger]): Logger for evaluation metrics.
        keep_best (bool): Whether to keep the best agent based on evaluation performance.
        eval_freq (int): Frequency of evaluation in terms of steps, iterations, or optimization steps.
        deterministic (bool): Whether to use a deterministic policy during evaluation.
    
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 num_episodes: int = 1,
                 logger: Optional[Logger] = None,
                 keep_best: bool = False,
                 eval_freq: int = 1,
                 deterministic: bool = False
                 ) -> None:
        super().__init__(eval_freq=eval_freq)
        self.env = env
        self.num_env = env.num_envs
        self.num_episodes = num_episodes
        self.logger = logger
        self.keep_best = keep_best
        self.deterministic = deterministic
        self.best_reward = float("-inf")
        self.best_agent = None

        self.collector = ParallelCollector(env)

    def evaluate(self, 
                 agent,
                 iteration: int,
                 is_last: bool = False
                 ) -> None:
        """
        Evaluate the agent's performance in the given environment.

        Args:
            agent: The agent to be evaluated.
            iteration (int): The current iteration number.
            is_last (bool): Whether this is the last evaluation.
        """
        # Check if evaluation should be performed
        if not is_last and not self._should_evaluate(iteration):
            return
        
        # Collect desired number of trajectories
        trajectories = self.collector.collect_trajectory(agent, num_trajectories=self.num_episodes)

        rewards = trajectories['reward'].detach().cpu().numpy().reshape(-1)
        dones = trajectories['done'].detach().cpu().numpy().reshape(-1)

        # Sum rewards for each episode
        episode_rewards = []
        running_reward = 0.0
        for reward, done in zip(rewards, dones):
            running_reward += reward
            if done:
                episode_rewards.append(running_reward)
                running_reward = 0.0

        # Calculate average reward across episodes
        avg_reward = np.mean(episode_rewards)

        # Update the best reward and agent if the current average reward is better
        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward

            if self.keep_best:
                self.best_agent = copy.deepcopy(agent)

        if self.logger is not None:
            self.logger.log_scalar("evaluation_reward", avg_reward, iteration=iteration)
            self.logger.log_scalar("evaluation_reward_std", np.std(episode_rewards), iteration=iteration)
            self.logger.log_scalar("evaluation_reward_max", np.max(episode_rewards), iteration=iteration)
            self.logger.log_scalar("evaluation_reward_min", np.min(episode_rewards), iteration=iteration)


    def close(self) -> None:
        """
        Close the evaluator and release any resources.
        """
        if self.keep_best and self.best_agent is not None and self.logger is not None:
            self.logger.save_agent(self.best_agent, "agent-best.pt")

class NumberOfStepsEvaluator(Evaluator):
    """
    Evaluator that evaluates the agent's performance to reach a minimum reward threshold within the lowest number of steps. This evaluator is intended to be used when an agent is able to achieve a maximum desired reward and you want to evaluate which agent learns the fastest.

    Args:
        env (EnvironmentInterface): The environment to evaluate the agent in.
        reward_threshold (float): The minimum reward threshold to achieve.
        num_episodes (int): The number of episodes to run for evaluation.
        logger (Optional[Logger]): Logger for evaluation metrics.
        keep_best (bool): Whether to keep the best agent based on evaluation performance.
        eval_freq (int): Frequency of evaluation in terms of steps, iterations, or optimization steps.
        deterministic (bool): Whether to use a deterministic policy during evaluation.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 reward_threshold: float,
                 num_episodes: int = 1,
                 logger: Optional[Logger] = None,
                 keep_best: bool = False,
                 eval_freq: int = 1,
                 deterministic: bool = False
                 ) -> None:
        super().__init__(eval_freq=eval_freq)
        self.env = env
        self.reward_threshold = reward_threshold
        self.num_episodes = num_episodes
        self.logger = logger
        self.keep_best = keep_best
        self.deterministic = deterministic
        self.best_agent = None
        self.best_timestep = math.inf

        self.collector = ParallelCollector(env)

    def evaluate(self,
                 agent,
                 iteration: int,
                 is_last: bool = False
                 ) -> None:
        """
        Evaluate the agent's performance in the given environment based on timesteps.

        Args:
            agent: The agent to be evaluated.
            iteration (int): The current iteration number.
            is_last (bool): Whether this is the last evaluation.

        Returns:
            None
        """
        # Check if evaluation should be performed
        if not is_last and not self._should_evaluate(iteration):
            return
        
        trajectories = self.collector.collect_trajectory(agent, num_trajectories=self.num_episodes)

        rewards = trajectories['reward'].detach().cpu().numpy().reshape(-1)
        dones = trajectories['done'].detach().cpu().numpy().reshape(-1)

        # Sum rewards for each episode
        episode_rewards = []
        running_reward = 0.0
        for reward, done in zip(rewards, dones):
            running_reward += reward
            if done:
                episode_rewards.append(running_reward)
                running_reward = 0.0

        # Calculate average reward across episodes
        avg_reward = np.mean(episode_rewards)

        # Check if the average reward meets the threshold and update best timestep
        if avg_reward >= self.reward_threshold and iteration < self.best_timestep:
            self.best_timestep = iteration

            if self.keep_best:
                self.best_agent = copy.deepcopy(agent)

        if self.logger is not None:
            self.logger.log_scalar("evaluation_numsteps", self.best_timestep, iteration=iteration)
        
    def close(self) -> None:
        """
        Close the evaluator and release any resources.
        """
        if self.keep_best and self.best_agent is not None and self.logger is not None:
            self.logger.save_agent(self.best_agent, "agent-best.pt")        