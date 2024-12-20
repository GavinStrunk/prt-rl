import torch

class QTable:
    r"""
    The Q table implements a matrix of state-action values.

    For example, if there are 3 states, 2 actions, and an initial value of 0.1 the Q table will look like:

    +------+-------+--------+
    |      | 0     | 1      |
    +=======================+
    | 0    | 0.1   | 0.1    |
    +------+-------+--------+
    | 1    | 0.1   | 0.1    |
    +------+-------+--------+
    | 2    | 0.1   | 0.1    |
    +------+-------+--------+

    Args:
        state_dim (int): Number of states
        action_dim (int): Number of actions
        initial_value (float): Initial value for the entire Q table. Default is 0.0.
        device (str): Device to use. Default is 'cpu'.

    Example:
        from prt_rl.utils.qtable import QTable

        qtable = QTable(state_dim=3, action_dim=2)
        qtable.update_q_value(state=1, action=3, q_value=0.1)
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 initial_value: float = 0.0,
                 device: str = 'cpu'
                 ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initial_value = initial_value
        self.device = device
        self.q_table = torch.zeros((self.state_dim, self.action_dim), dtype=torch.float32, device=device) + initial_value

    def get_action_values(self,
                          state: int
                          ) -> torch.Tensor:
        """
        Returns the state action values for a given state.

        Args:
            state (int): state value to get action values for

        Returns:
            torch.tensor: action values for given state
        """
        return self.q_table[state]

    def get_state_action_value(self,
                               state: int,
                               action: int
                               ) -> torch.Tensor:
        """
        Returns the value for the given state-action pair.

        Args:
            state (int): state value to get the value for
            action (int): action value to get the value for

        Returns:
            torch.tensor: value for the given state-action pair. This is a scalar value with size torch.Size([]).
        """
        return self.q_table[state][action]

    def update_q_value(self,
                       state: int,
                       action: int,
                       q_value: float
                       ) -> None:
        """
        Updates the Q table for a given state-action pair with given q-value.
        Args:
            state:
            action:
            q_value:

        Returns:

        """
        self.q_table[state, action] = q_value