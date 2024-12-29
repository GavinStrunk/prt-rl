from abc import ABC, abstractmethod
from tensordict.tensordict import TensorDict
from typing import Optional
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.utils.policies import Policy, QNetworkPolicy
from prt_rl.utils.loggers import Logger


class TDTrainer(ABC):
    """
    Temporal Difference Reinforcement Learning (TD) trainer base class. RL algorithms are implementations of this class.

    """
    def __init__(self,
                 env: EnvironmentInterface,
                 policy: Policy,
                 logger: Optional[Logger] = None,
                 ) -> None:
        self.env = env
        self.policy = policy
        self.logger = logger or Logger()

    @abstractmethod
    def update_policy(self,
                      experience: TensorDict,
                      ) -> None:
        raise NotImplementedError

    def get_policy(self) -> Policy:
        """
        Returns the current policy.

        Returns:
            Policy: current policy object.
        """
        return self.policy

    def train(self,
              num_episodes: int,
              ) -> None:

        cumulative_reward = 0
        for i in range(num_episodes):
            obs_td = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action_td = self.policy.get_action(obs_td)
                obs_td = self.env.step(action_td)
                self.update_policy(obs_td)
                episode_reward += obs_td['next','reward']
                done = obs_td['next', 'done']
                obs_td = self.env.step_mdp(obs_td)

            cumulative_reward += episode_reward
            print(f"Episode {i} Reward: {episode_reward}")
            self.logger.log_scalar('episode_reward', episode_reward, iteration=i)
            self.logger.log_scalar('cumulative_reward', cumulative_reward, iteration=i)

        self.logger.close()


class ANNTrainer(TDTrainer):
    def __init__(self,
                 env: EnvironmentInterface,
                 policy: QNetworkPolicy,
                 ) -> None:
        super().__init__(env, None)
        self.policy = policy

    def get_policy_network(self):
        return self.policy.q_network

    @abstractmethod
    def update_policy(self,
                      experience: TensorDict,
                      ) -> None:
        raise NotImplementedError