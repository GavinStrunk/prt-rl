import numpy as np
from prt_rl.exact.policies import epsilon_greedy

class SARSA:
    r"""
    SARSA algorithm.

    .. math::
        Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]

    """

    def __init__(self,
                 num_states,
                 num_actions,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def select_action(self, state) -> int:
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

    def learn(self, trajectory: tuple) -> int:
        """
        Updates the q_table values.

        Args:
            trajectory (tuple): (s_t, a_t, r_{t+1}, s_{t+1}) tuple

        Returns:
            int: A_{t+1}
        """
        # Choose A_{t+1} from policy based on S_{t+1}
        st, at, rt1, st1 = trajectory
        at1 = self.select_action(st1)
        self.q_table[st][at] += self.alpha * (rt1 + self.gamma*self.q_table[st1][at1] - self.q_table[st][at])

        return at1


class ExpectedSARSA(SARSA):
    """
    Expected SARSA algorithm.

    """
    def __init__(self,
                 num_states,
                 num_actions,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 ):
        super(ExpectedSARSA, self).__init__(num_states, num_actions, alpha, gamma, epsilon)

    def learn(self, trajectory: tuple) -> None:
        """
        Updates the q_table values.
        Args:
            trajectory:

        Returns:

        """
        st, at, rt1, st1 = trajectory
        actions = self.q_table[st1][:]
        a_stars = np.where(actions == np.max(actions))[0]
        a_star = a_stars[0]
        expect_a = 0
        for i in range(self.num_actions):
            a_val = self.q_table[st1][i]
            if i == a_star:
                prob_a = (1 - self.epsilon) / len(a_stars) + self.epsilon / self.num_actions
            else:
                prob_a = self.epsilon / self.num_actions

            expect_a += prob_a*a_val

        self.q_table[st][at] += self.alpha * (rt1 + self.gamma*expect_a - self.q_table[st][at])