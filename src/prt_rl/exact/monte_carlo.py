import numpy as np
from prt.rl.exact.policies import epsilon_greedy

class MonteCarlo:
    r"""
    On-policy First Visit Monte Carlo Algorithm

    .. math::
        \begin{equation}
        Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \frac{1}{N}[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1} - Q(S_t,A_t)] \\
        q_s \leftarrow q_s + \frac{1}{n}[G_t - q_s]
        \end{equation}

    Args:
        num_states (int): number of environment states
        num_actions (int): number of agent actions
        epsilon (float): probability of choosing a random action
        gamma (float): discount factor
    """
    def __init__(
            self,
            num_states: int,
            num_actions: int,
            epsilon: float=0.1,
            gamma: float=1.0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma

        # Initialize the Q-table and visit table
        self.q_table = np.zeros((num_states, num_actions))
        self.visit_table = np.zeros((num_states, num_actions))

    def select_action(self, state) -> int:
        """
        Selects an action based on the epsilon greedy policy

        Args:
            state (int): current state of the environment

        Returns:
            int: action to take
        """
        actions = self.q_table[state]
        action = epsilon_greedy(actions, epsilon=self.epsilon)

        return action

    def learn(self, trajectory: list[tuple]) -> None:
        """

        Args:
            trajectory (list[tuple]): Episode trajectory as a list of tuples (reward, state, action)

        Returns:
            None
        """
        # Initialize the return to zero
        G = 0

        # Learn by working through the trajectory backwards
        for t in reversed(range(len(trajectory) - 1)):
            _, state, action = trajectory[t]
            reward, _, _ = trajectory[t+1]
            # Update the Return
            G = self.gamma * G + reward

            # If this is a first visit update the visit and q tables
            if self.is_first_visit(trajectory, t):
                self.update_visit(state, action)
                self.update_q(state, action, G)

    def is_first_visit(self, trajectory: list[tuple], t: int) -> bool:
        """
        Checks if the state,action pair at timestep t in the trajectory is the first visit

        Args:
            trajectory (list[tuple]): Episode trajectory as a list of tuples (r, s, a)
            t (int): Current timestep

        Returns:
            bool: True or False
        """
        if t > len(trajectory) - 1:
            raise IndexError("Index is outside the bounds of the trajectory")

        sa_sets = []
        pairs = set()

        for r, s, a in trajectory:
            # Create state,action sets
            pairs.add((s, a))

            # Append the set to the list of sets
            sa_sets.append(pairs.copy())

        _, state, action = trajectory[t]

        # If t is 0 this has to be the first visit
        if t > 0:
            return (state, action) not in sa_sets[t-1]
        else:
            return True

    def update_visit(self, state: int, action: int) -> None:
        """
        Updates the visit table by incrementing the value of the given state,action pair

        Args:
            state (int): Current state
            action (int): Selected action

        Returns:
            None
        """
        self.visit_table[state][action] += 1

    def update_q(self, state: int, action: int, G: float) -> None:
        """
        Updates the q table value by averaging the latest return with the previous returns at a given state,action pair

        Args:
            state (int): Current state
            action (int): Selected action
            G (int): Current return

        Returns:
            None
        """
        n = self.visit_table[state][action]
        self.q_table[state][action] += 1/n*(G - self.q_table[state][action])