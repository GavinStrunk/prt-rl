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
        batch_size (int): Batch size (number of environments).
        initial_value (float): Initial value for the entire Q table. Default is 0.0.
        track_visits (bool): If True, a Visit table will be created to track state-action visits. Default is False.
        device (str): Device to use. Default is 'cpu'.

    Example:
        from prt_rl.utils.qtable import QTable

        qtable = QTable(state_dim=3, action_dim=2)
        qtable.update_q_value(state=1, action=3, q_value=0.1)
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 batch_size: int = 1,
                 initial_value: float = 0.0,
                 track_visits: bool = False,
                 device: str = 'cpu'
                 ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.initial_value = initial_value
        self.track_visits = track_visits
        self.device = device
        self.q_table = torch.zeros((self.batch_size, self.state_dim, self.action_dim), dtype=torch.float32, device=device) + initial_value

        # Initialize the visit table if they are being kept
        if self.track_visits:
            self.visit_table = torch.zeros((self.batch_size, self.state_dim, self.action_dim), dtype=torch.float32, device=device)

    def init_args(self) -> dict:
        """
        Returns the initialization arguments that are required to construct the QTable.

        Returns:
            dict: Initialization arguments
        """
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'batch_size': self.batch_size,
            'initial_value': self.initial_value,
            'track_visits': self.track_visits,
            'device': self.device
        }

    def get_action_values(self,
                          state: torch.Tensor
                          ) -> torch.Tensor:
        """
        Returns the state action values for a given state.

        Args:
            state (torch.Tensor): state value to get action values for with shape (# env, 1)

        Returns:
            torch.Tensor: action values for given state with shape (# env, # actions)
        """
        assert state.dtype == torch.int, "State values must be integers."
        state = state.squeeze(-1)
        return self.q_table[torch.arange(self.q_table.size(0)), state]

    def get_state_action_value(self,
                               state: torch.Tensor,
                               action: torch.Tensor
                               ) -> torch.Tensor:
        """
        Returns the value for the given state-action pair.

        Args:
            state (torch.Tensor): state value to get the value for with shape (# env, 1)
            action (torch.Tensor): action value to get the value for with shape (# env, 1)

        Returns:
            torch.Tensor: value for the given state-action pair with shape (# env, 1)
        """
        assert state.dtype == torch.int, "State values must be integers."
        assert action.dtype == torch.int, "Action values must be integers."
        state = state.squeeze(-1)
        action = action.squeeze(-1)
        return self.q_table[torch.arange(self.q_table.size(0)), state, action].unsqueeze(-1)

    def get_visit_count(self,
                        state: torch.Tensor,
                        action: torch.Tensor
                        ) -> torch.Tensor:
        """
        Returns the number of visits for a given state-action pair.

        Args:
            state (torch.Tensor): state value to get the number of visits for with shape (# env, 1)
            action (torch.Tensor): action value to get the number of visits for with shape (# env, 1)

        Returns:
            torch.Tensor: number of visits for given state-action pair with shape (# env, 1))
        """
        assert state.dtype == torch.int, "State values must be integers."
        assert action.dtype == torch.int, "Action values must be integers."
        state = state.squeeze(-1)
        action = action.squeeze(-1)
        return self.visit_table[torch.arange(self.visit_table.size(0)), state, action].unsqueeze(-1)

    def update_q_value(self,
                       state: torch.Tensor,
                       action: torch.Tensor,
                       q_value: torch.Tensor
                       ) -> None:
        """
        Updates the Q table for a given state-action pair with given q-value.
        Args:
            state (torch.Tensor): state value to update the Q table for with shape (# env, 1)
            action (torch.Tensor): action value to update the Q table for with shape (# env, 1)
            q_value (torch.Tensor): q-value to update the Q table for with shape (# env, 1)

        """
        assert state.dtype == torch.int, "State values must be integers."
        assert action.dtype == torch.int, "Action values must be integers."

        state = state.squeeze(-1)
        action = action.squeeze(-1)
        q_value = q_value.squeeze(-1)

        # Use advanced indexing to update the q-table
        self.q_table[torch.arange(self.q_table.size(0)), state, action] = q_value

    def update_visits(self,
                      state: torch.Tensor,
                      action: torch.Tensor
                      ) -> None:
        """
        Updates the Visit table for a given state-action pair.

        Args:
            state (torch.Tensor): state value to update the Visit table for with shape (# env, 1)
            action (torch.Tensor): action value to update the Visit table for with shape (# env, 1)
        """
        assert state.dtype == torch.int, "State values must be integers."
        assert action.dtype == torch.int, "Action values must be integers."
        state = state.squeeze(-1)
        action = action.squeeze(-1)
        self.visit_table[torch.arange(self.visit_table.size(0)), state, action] += 1.0