import torch
from prt_rl.model_based.planners.rollout import rollout_action_sequence

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
                 num_action_sequences: int = 100,
                 device: str = 'cpu'
                 ) -> None:
        assert action_mins.shape == action_maxs.shape, "Action mins and maxs must have the same shape."
        assert action_mins.ndim == 2 and action_mins.shape[1] == 1, "Expected shape (action_dim, 1)"
        self.device = torch.device(device)
        self.action_mins = action_mins.to(self.device)
        self.action_maxs = action_maxs.to(self.device)
        self.planning_horizon = planning_horizon
        self.num_action_sequences = num_action_sequences

    def _get_action_sequence(self, num_action_sequences: int) -> torch.Tensor:
        """
        Generate a batch of random action sequences.

        Args:
            num_action_sequences (int): Number of random action sequences to generate.

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

    def plan(self, 
             model_fcn: Callable, 
             model_config: Any, 
             reward_fcn: Callable, 
             state: torch.Tensor
             ) -> torch.Tensor:
        # Plan action sequences
        num_action_seq = self.num_action_sequences

        # Random plan action sequences with shape (B, H, action_dim)
        action_sequences = self._get_action_sequence(num_action_seq)
        
        # Evaluate action sequences using the model and reward function
        rollout = rollout_action_sequence(model_config, model_fcn, state, action_sequences)
        
        # Evaluate the objective value (reward) for each action sequence
        rewards = reward_fcn(rollout['state'], rollout['action'], rollout['next_state']) 

        # Sort by reward value
        max_index = torch.argmax(rewards)

        # Return the first action from the best action sequence
        return action_sequences[max_index.item(), 0, :].unsqueeze(0)
    