from abc import ABC, abstractmethod
from tensordict import tensordict

from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.utils.policies import Policy

class TDTrainer(ABC):
    """
    Temporal Difference Reinforcement Learning (TD) trainer base class. RL algorithms are implementations of this class.

    """
    def __init__(self,
                 env_params: EnvParams,
                 env: EnvironmentInterface,
                 policy: Policy,
                 ) -> None:
        self.env_params = env_params
        self.env = env
        self.policy = policy

    @abstractmethod
    def update_policy(self,
                      trajectory: tensordict,
                      ) -> None:
        raise NotImplementedError

    def train(self,
              num_episodes: int,
              ) -> None:

        obs_td = self.env.reset()
        for _ in range(num_episodes):
            action_td = self.policy.get_action(obs_td)
            obs_td = self.env.step(action_td)
            self.update_policy(obs_td)



