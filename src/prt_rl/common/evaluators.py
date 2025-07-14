from abc import ABC, abstractmethod
import copy
from typing import Optional
import numpy as np
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.loggers import Logger

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

class MaxRewardEvaluator(Evaluator):
    """
    Base class for all evaluators in the PRT-RL framework.
    Evaluators are used to assess the performance of agents or policies.
    This class assumes that the caller calls evaluate at the desired frequency.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 num_episodes: int = 1,
                 logger: Optional[Logger] = None,
                 keep_best: bool = False,
                 ) -> None:
        self.env = env
        self.num_episodes = num_episodes
        self.logger = logger
        self.keep_best = keep_best
        self.best_reward = float("-inf")
        self.best_agent = None

    def evaluate(self, 
                 agent,
                 iteration: int
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
        rewards = []
        for i in range(self.num_episodes):
            state, _ = self.env.reset(seed=i)

            episode_reward = 0.0
            done = False

            while not done:
                state = state.float()
                action = agent(state)

                next_state, reward, done, _ = self.env.step(action)

                episode_reward += reward.item()
                state = next_state

            rewards.append(episode_reward)

        avg_reward = np.mean(rewards)
        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward

            if self.keep_best:
                self.best_agent = copy.deepcopy(agent)

        if self.logger is not None:
            self.logger.log_scalar("evaluation_reward", avg_reward, iteration=iteration)
            self.logger.log_scalar("evaluation_reward_std", np.std(rewards), iteration=iteration)

    def close(self) -> None:
        """
        Close the evaluator and release any resources.
        """
        if self.keep_best and self.best_agent is not None and self.logger is not None:
            self.logger.save_agent(self.best_agent, "agent-best.pt")