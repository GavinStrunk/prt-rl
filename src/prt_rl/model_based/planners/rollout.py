import torch
from typing import Any

def rollout_action_sequence(
        model_config: Any, 
        model_fcn: callable, 
        initial_state: torch.Tensor, 
        action_sequence: torch.Tensor
        ):
    """
    Rollout a sequence of actions using the provided model starting from the initial state.

    Args:
        model: The model used to predict the next state.
        initial_state (torch.Tensor): The initial state of the environment.
        action_sequence (torch.Tensor): A sequence of actions to be taken.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the states, actions, and next_states encountered during the rollout.
            - 'state': Tensor of shape (B, T, state_dim) containing the states at each time step.
            - 'action': Tensor of shape (B, T, action_dim) containing the actions taken at each time step.
            - 'next_state': Tensor of shape (B, T, state_dim) containing the next states at each time step.
    """
    B, T, _ = action_sequence.shape

    # Repeat the initial state to match the number of batch action sequences - (B, state_dim)
    state = initial_state.repeat(B, 1)

    states = [state]
    actions = []
    next_states = []
    for t in range(T):
        # Get the action at time step t - (B, action_dim)
        action = action_sequence[:, t, :]
        actions.append(action)

        # Predict the next state using the model
        next_state = model_fcn(model_config, state, action)
        next_states.append(next_state)

        # Update the current state
        state = next_state

    return {
        'state': torch.stack(states, dim=1),          # (B, T, state_dim)
        'action': torch.stack(actions, dim=1),        # (B, T, action_dim)
        'next_state': torch.stack(next_states, dim=1) # (B, T, state_dim)
    }