from typing import Optional, Union, List, Tuple
import numpy as np
from prt_sim.jhu.base import BaseEnvironment

class KArmBandits(BaseEnvironment):
    """
    K-arm Bandits simulation

    The k-arm bandit problem chooses the true value $q_*(a)$ of each of the actions according to a normal distribution with mean zero and unit variance. The actual rewards are selected according to a mean $q*(a)$ and unit variance normal distribution. They chose average reward vs steps and percent optimal action vs steps as the metrics to track.

    Args:
        num_bandits (int): Number of random bandits

    References:
        [1] Sutton, Barto: Introduction to Reinforcement Learning Edition 2, p29

    Examples:

    """
    def __init__(self,
                 num_bandits: int = 10,

                 ) -> None:
        assert num_bandits > 0, "Number of bandits must be greater than 0"
        self.num_bandits = num_bandits
        self.bandit_probs = np.zeros(self.num_bandits)

    def get_number_of_states(self) -> int:
        """
        Returns the number of states

        Returns:
            int: number of states
        """
        return 0

    def get_number_of_actions(self) -> int:
        """
        Returns the number of actions which is equal to the number of bandits

        Returns:
            int: number of actions
        """
        return self.num_bandits



    def reset(self,
              seed: Optional[int] = None,
              randomize_start: Optional[bool] = False
              ) -> int:
        """
        Resets the bandits probabilities randomly or with provided values.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            randomize_start (bool, optional): Whether to randomize the starting state. Not all environments will support this. Defaults to False.

        Returns:
            int: current state value
        """
        assert not randomize_start, "Randomizing starting state is not supported"

        if seed is not None:
            np.random.seed(seed)

        self.bandit_probs = np.random.normal(0, 1.0, size=self.num_bandits)
        return 0

    def execute_action(self,
                       action: int
                       ) -> Tuple[int, float, bool]:
        """
        Executes the action and a step in the environment.

        Args:
            action (int): bandit to play

        Returns:
            tuple: (state, reward, done) the reward is the only relevant value
        """
        assert self.num_bandits-1 >= action >= 0, "Action must be in the interval [0, number of bandits - 1]."
        # There is no state or episode for bandits just a single play
        reward = np.random.normal(self.bandit_probs[action], 1.0)
        return 0, reward, True

    def get_optimal_bandit(self) -> int:
        """
        Returns the optimal bandit. This should not be used by the agent, but only for evaluation purposes.

        Returns:
            int: optimal bandit index
        """
        return np.argmax(self.bandit_probs)

