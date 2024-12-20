import numpy as np
from prt_rl.utils.policies import epsilon_greedy

from tensordict.tensordict import TensorDict
import torch
from typing import Optional
from prt_rl.exact.trainers import TDTrainer
from prt_rl.utils.decision_functions import DecisionFunction, EpsilonGreedy
from prt_rl.utils.policies import QTablePolicy
from prt_rl.utils.qtable import QTable

class QLearning(TDTrainer):
    r"""
    Q-Learning trainer.

    .. math::
        Q(s,a)

    Args:
        env_params (EnvParams): environment parameters.
    """
    def __init__(self,
                 env_params,
                 decision_function: Optional[DecisionFunction] = None,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 ) -> None:
        self.deterministic = False
        self.gamma = gamma
        self.alpha = alpha

        if decision_function is None:
            decision_function = EpsilonGreedy(epsilon=0.1)

        self.q_table = QTable(
                state_dim=env_params.observation_shape,
                action_dim=env_params.action_shape,
                initial_value=0
            )

        policy = QTablePolicy(
            q_table=self.q_table,
            decision_function=decision_function
        )
        super().__init__(env_params, policy)

    def update_policy(self, trajectory: TensorDict) -> None:
        st = trajectory["observation"]
        at = trajectory["action"]
        st1 = trajectory['next','observation']
        rt1 = trajectory['next', 'reward']

        # Get the action with the maximum value for the next state
        q_max = torch.max(self.q_table.get_action_values(st1))

        # Update Q table
        if self.deterministic:
            qval_update = rt1 + self.gamma*q_max
        else:
            qval = self.q_table.get_state_action_value(st, at)
            qval_update = qval + self.alpha * (rt1 + self.gamma*q_max - qval)

        self.q_table.update_q_value(state=st, action=at, q_value=qval_update)



class QLearning2:
    r"""
    Q Learning algorithm


    """
    def __init__(self,
                 num_states,
                 num_actions,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 deterministic=False,
                 ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.deterministic = deterministic
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def select_action(self, state):
        """
        Selects an action based on the epsilon-greedy policy.

        Args:
            state (int): current state of the environment

        Returns:
            int: action to take
        """
        actions = self.q_table[state]
        action = epsilon_greedy(actions, epsilon=self.epsilon)

        return action

    def learn(self, trajectory: tuple) -> None:
        """

        Args:
            trajectory (tuple): (state, action, reward, next_state)

        Returns:

        """
        st, at, rt1, st1 = trajectory
        q_max = np.max(self.q_table[st1][:])
        if self.deterministic:
            self.q_table[st][at] = rt1 + self.gamma*q_max
        else:
            self.q_table[st][at] += self.alpha * (rt1 + self.gamma*q_max - self.q_table[st][at])

