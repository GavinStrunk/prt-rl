import torch

class QTable:
    def __init__(self,
                 state_dim,
                 action_dim,
                 initial_value,
                 device: str = 'cpu'
                 ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initial_value = initial_value
        self.device = device
        self.q_table = torch.zeros((self.state_dim, self.action_dim), dtype=torch.float32, device=device) + initial_value

    def get_action_values(self,
                          state: int) -> torch.Tensor:
        """
        Returns the state action values for a given state.
        Args:
            state:

        Returns:

        """
        return self.q_table[state]

    def update_table(self,
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