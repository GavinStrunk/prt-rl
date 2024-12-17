import numpy as np

class KArmBandits:
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
                 device: str = 'cpu'
                 ) -> None:
        assert num_bandits > 0, "Number of bandits must be greater than 0"
        self.num_bandits = num_bandits
        self.device = device
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

    def get_optimal_bandit(self) -> int:
        """
        Returns the optimal bandit. This should not be used by the agent, but only for evaluation purposes.

        Returns:
            int: optimal bandit index
        """
        return np.argmax(self.bandit_probs)

    def reset(self, initial_state: list[float] = None) -> None:
        """
        Resets the bandits probabilities randomly or with provided values.

        Args:
            initial_state (list[float]): list of bandit probabilities

        Returns:
            None
        """
        if initial_state is None:
            self.bandit_probs = np.random.normal(0, 1.0, size=self.num_bandits)
        else:
            assert len(initial_state) == self.num_bandits, "Number of initial probabilities must match the number of bandits."
            self.bandit_probs = np.array(initial_state)

    def execute_action(self, action: int) -> tuple:
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
        return None, reward, True

