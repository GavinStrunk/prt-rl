import torch


class RandomShootingPlanner:
    """
    Random Shooting Planner for Model-Based Reinforcement Learning.
    
    This planner samples random action sequences and selects the best one based on a given objective function.

    Args:
        action_mins (torch.Tensor): Minimum values for each action dimension. Shape (action_dim, 1).
        action_maxs (torch.Tensor): Maximum values for each action dimension. Shape (action_dim, 1).
        planning_horizon (int): Number of steps to plan ahead.
    """
    def __init__(self,
                 action_mins: torch.Tensor,
                 action_maxs: torch.Tensor,
                 planning_horizon: int = 10,
                 device: str = 'cpu'
                 ) -> None:
        assert action_mins.shape == action_maxs.shape, "Action mins and maxs must have the same shape."
        assert action_mins.ndim == 2 and action_mins.shape[1] == 1, "Expected shape (action_dim, 1)"
        self.device = torch.device(device)
        self.action_mins = action_mins.to(self.device)
        self.action_maxs = action_maxs.to(self.device)
        self.planning_horizon = planning_horizon

    def plan(self, num_action_sequences: int) -> torch.Tensor:
        """
        Plan a sequence of actions using random shooting.

        Args:
            num_action_sequences (int): Number of random action sequences to sample.

        Returns:
            torch.Tensor: A tensor of shape (B, H, A) where:
                B = num_action_sequences
                H = planning_horizon
                A = action_dim
        """
        action_dim = self.action_mins.shape[0]
        B, H, A = num_action_sequences, self.planning_horizon, action_dim

        # (B, H, A) uniform in [0, 1]
        random_uniform = torch.rand(B, H, A, device=self.device)

        # (A,) ranges
        low = self.action_mins.squeeze(-1)  # (A,)
        high = self.action_maxs.squeeze(-1)  # (A,)
        range_ = high - low  # (A,)

        # Reshape to broadcast over (B, H, A)
        low = low.view(1, 1, A)
        range_ = range_.view(1, 1, A)

        # Uniform sampling in [low, high] per action dim
        action_seqs = low + random_uniform * range_

        return action_seqs
    