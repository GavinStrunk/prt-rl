import numpy as np
from prt.rl.exact.policies import epsilon_greedy

class SampleAverage:
    """
    Sample Average method estimates the value of action by estimating the average sample of relevant rewards.

    """
    def __init__(self,
                 num_actions: int,
                 epsilon: float,
                 ) -> None:
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(self.num_actions)
        self.selections = np.zeros(self.num_actions)

    def select_action(self) -> int:
        """
        Selects the best action.

        Returns:

        """
        action = epsilon_greedy(self.q_values, self.epsilon)
        return action

    def learn(self, action: int, reward: float) -> None:
        """
        Updates the visit list and the average action value

        Args:
            action (int): Selected action that was taken
            reward (float): Reward received

        Returns:
            None
        """
        self.selections[action] += 1
        # Incremental average update
        self.q_values[action] += 1/(self.selections[action]) * (reward - self.q_values[action])
