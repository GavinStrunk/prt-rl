from abc import ABC, abstractmethod
import copy
from typing import Optional
import numpy as np
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.collectors import get_action_from_policy, ParallelCollector

class Evaluator(ABC):
    """
    Base class for all evaluators in the PRT-RL framework.
    This class provides a common interface for evaluating agents in different environments.
    """
    def evaluate(self, agent, iteration: int) -> None:
        """
        Evaluate the agent's performance in the given environment.

        Args:
            agent: The agent to be evaluated.
            iteration (int): The current iteration number.

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
        self.env = env
        self.num_env = env.num_envs
        self.num_episodes = num_episodes
        self.logger = logger
        self.keep_best = keep_best
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.last_evaluation_iteration = 0
        self.best_reward = float("-inf")
        self.best_agent = None

        self.collector = ParallelCollector(env)

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

    def evaluate(self, 
                 agent,
                 iteration: int,
                 is_last: bool = False
                 ) -> None:
        """
        Evaluate the agent's performance in the given environment.

        Args:
            agent: The agent to be evaluated.
            env: The environment in which to evaluate the agent.
            num_episodes: The number of episodes to run for evaluation.

        Returns:
            A dictionary containing evaluation metrics.
        """
        # Check if evaluation should be performed
        if not is_last and not self._should_evaluate(iteration):
            return
        
        trajectories = self.collector.collect_trajectory(agent, num_trajectories=self.num_episodes)
        rewards = [t['rewards'] for t in trajectories]

        avg_reward = np.mean(rewards)
        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward

            if self.keep_best:
                self.best_agent = copy.deepcopy(agent)

        if self.logger is not None:
            self.logger.log_scalar("evaluation_reward", avg_reward, iteration=iteration)
            self.logger.log_scalar("evaluation_reward_std", np.std(rewards), iteration=iteration)
            self.logger.log_scalar("evaluation_reward_max", np.max(rewards), iteration=iteration)
            self.logger.log_scalar("evaluation_reward_min", np.min(rewards), iteration=iteration)


    def close(self) -> None:
        """
        Close the evaluator and release any resources.
        """
        if self.keep_best and self.best_agent is not None and self.logger is not None:
            self.logger.save_agent(self.best_agent, "agent-best.pt")