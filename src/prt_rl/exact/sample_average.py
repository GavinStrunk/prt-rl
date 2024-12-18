import numpy as np

from prt_rl.env.interface import EnvParams
from prt_rl.utils.policies import epsilon_greedy, QTablePolicy
from prt_rl.utils.qtable import QTable
from prt_rl.utils.decision_functions import EpsilonGreedy

from tensordict import TensorDict
from prt_rl.exact.trainers import TDTrainer

class SampleAverage(TDTrainer):
    def __init__(self,
                 env_params: EnvParams,
                 epsilon: float,
                 ):
        policy = QTablePolicy(
            q_table=QTable(
                state_dim=env_params.observation_shape,
                action_dim=env_params.action_shape,
                initial_value=0
            ),
            decision_function=EpsilonGreedy(
                epsilon=epsilon,
            ),
        )
        super().__init__(env_params, policy)

    def update_policy(self, trajectory: TensorDict) -> None:
        action = trajectory.get('next', 'action')
        reward = trajectory.get('next','reward')

        pass


class SampleAverage2:
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
