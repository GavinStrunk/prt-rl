import numpy as np
from prt.rl.exact.policies import epsilon_greedy


class QLearning:
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

