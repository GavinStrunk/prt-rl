"""
Utility functions for reinforcement learning agents that are used across different algorithms.
"""
import random
import numpy as np
import torch
from typing import Tuple, Optional

def set_seed(seed: int):
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # (Optional) Determinism in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)  # Uncomment for stricter control (may raise errors)

def polyak_update(target: torch.nn.Module, network: torch.nn.Module, tau: float) -> None:
    """
    Updates a target network using Polyak averaging.

    When tau is 0 the target is unchanged and when tau is 1 a hard update is performed. The parameters of the target network are updated in place.

    .. math::
        \Theta_{target} = \tau * \Theta_{\pi} + (1 - \tau) * \Theta_{target}

    Args:
        target (torch.nn.Module): The target network to be updated.
        network (torch.nn.Module): The policy network from which parameters are taken.
        tau (float): The interpolation factor, typically in the range [0, 1].

    References:
    [1] https://github.com/DLR-RM/stable-baselines3/issues/93
    """
    # Perform Polyak update on parameters
    # for target_params, policy_params in zip(target.parameters(), network.parameters()):
    #     target_params.data.copy_(tau * policy_params.data + (1 - tau) * target_params.data)

    # Perform polyak update on state_dict which support parameters and buffers, but is slower than the above method
    target_sd = target.state_dict()
    source_sd = network.state_dict()

    for key in target_sd:
        target_sd[key].copy_(tau * source_sd[key] + (1 - tau) * target_sd[key])

def hard_update(target: torch.nn.Module, network: torch.nn.Module) -> None:
    """
    Updates a target network with the parameters of the proided network. 
    
    This is a hard update where the parameters are directly copied from the network to the target. The parameters of the target network are updated in place.

    .. math::
        \Theta_{target} = \Theta_{\pi}

    Args:
        target (torch.nn.Module): The target network to be updated.
        network (torch.nn.Module): The policy network from which parameters are taken.
    """
    target.load_state_dict(network.state_dict())

def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """
    Normalizes advantages to have zero mean and unit variance.

    Args:
        advantages (torch.Tensor): The advantages to normalize.

    Returns:
        torch.Tensor: The normalized advantages.
    """
    mean = advantages.mean()
    std = advantages.std()
    normalized_advantages = (advantages - mean) / (std + 1e-8)
    return normalized_advantages

def generalized_advantage_estimates(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (GAE) computes an advantage estimation that balances bias and variance.

    The GAE is defined as:

    .. math::
        A_t = \sum_{t'=t}^{\infty} (\gamma \lambda)^{t'-t} \delta_{t'}

    where :math:`\delta_{t'} = r_t + \gamma V(s_{t+1}) - V(s_t)`.

    When lambda is set to 1, this reduces to the Monte Carlo estimate of the advantage. When lambda is set to 0, it reduces to the one-step TD error.

    Args:
        rewards (torch.Tensor): Rewards from rollout with shape (T, N, 1) or (B, 1)
        values (torch.Tensor): Estimated state values with shape (T, N, 1) or (B, 1)
        dones (torch.Tensor): Done flags (1 if episode ended at step t, else 0) with shape (T, N, 1) or (B, 1)
        last_values (torch.Tensor): Value estimates for final state (bootstrap) with shape (N, 1)
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Estimated advantages with shape matching rewards shape
            - TD(lambda) returns with shape matching rewards shape
    """
    # Case 1: flattened batch (B, 1)
    if rewards.ndim == 2:
        B, _ = rewards.shape
        rewards = rewards.unsqueeze(1)  # (B, 1) → (B, 1, 1)
        values = values.unsqueeze(1)
        dones = dones.unsqueeze(1)
        last_values = last_values.unsqueeze(0).unsqueeze(1)  # (1, 1, 1)

        T, N = B, 1  # fake time-batch
        reshape_back = True

    # Case 2: time-major (T, N, 1)
    elif rewards.ndim == 3:
        T, N, _ = rewards.shape
        last_values = last_values.unsqueeze(0)  # (1, N, 1)
        reshape_back = False

    else:
        raise ValueError(f"Unsupported shape: {rewards.shape}")

    # Append last value for V(s_{t+1})
    values = torch.cat([values, last_values], dim=0)  # (T+1, N, 1)

    advantages = torch.zeros((T, N, 1), dtype=values.dtype, device=values.device)
    last_gae_lam = torch.zeros((N, 1), dtype=values.dtype, device=values.device)

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last_gae_lam = delta + gamma * gae_lambda * not_done * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values[:-1]  # TD(lambda) return

    if reshape_back:
        return advantages.squeeze(1), returns.squeeze(1)
    else:
        return advantages, returns

def rewards_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    last_values: Optional[torch.Tensor] = None,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Computes the discounted rewards-to-go returns for a batch of trajectories. This function supports bootstrapping partial trajectories, as well as, flattened or time-major inputs.

    The bootstrapped discounted rewards-to-go is defined as:

    .. math::
        G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r(s_{i,t'}, a_{i,t'}) + \gamma^{T-t} V(s_{i,T})

    where :math:`r(s_{i,t'}, a_{i,t'})` is the reward at time step :math:`t'`.

    Args:
        rewards (torch.Tensor): Rewards from rollout with shape (T, N, 1) or (B, 1)
        dones (torch.Tensor): Done flags (1 if episode ended at step t, else 0) with shape (T, N, 1) or (B, 1)
        last_values (Optional[torch.Tensor]): Value estimates for final state (bootstrap) with shape (N, 1). This is required if the last state is not terminal or 0 is assumed for the last value.
        gamma (float): Discount factor

    Returns:
        torch.Tensor: The rewards-to-go with shape that matches the input rewards shape.
    """
    if rewards.shape != dones.shape:
        raise ValueError(f"`rewards` and `dones` must match shape. Got {rewards.shape} and {dones.shape}")
    
    # Save the original shape so we can reshape the output
    original_shape = rewards.shape

    # Case 1: time-major (T, N, 1)
    if rewards.dim() == 3:
        # Reshape rewards and dones to (T, N)
        T, N, _ = rewards.shape
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)

        # If last_values is None, initialize running return to zero
        if last_values is None:
            running_return = torch.zeros(N, device=rewards.device)
        else:
            running_return = last_values.squeeze(-1)

    # Case 2: flattened batch (B, 1)
    elif rewards.dim() == 2:
        # Treat as as single environment where T=B and N=1
        T, N = rewards.shape

        # If last_values is None, initialize running return to zero
        if last_values is None:
            running_return = torch.zeros(N, device=rewards.device)
        else:
            running_return = last_values.view(-1)
    else:
        raise ValueError(f"Unsupported input shape: {rewards.shape}")

    rtg = torch.zeros_like(rewards)

    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        rtg[t] = running_return

    return rtg.view(original_shape)

def trajectory_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    last_values: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Computes the discounted returns for a sequence of rewards, also known as total discounted return.

    ..math ::
        \sum_{t'=0}^{T-1}\gamma^{t'}r(s_{i,t'},a_{i,t'})

    Args:
        rewards (torch.Tensor): Rewards from rollout with shape (T, N, 1)
        dones (torch.Tensor): Done flags (1 if episode ended at step t, else 0) with shape (T, N, 1)
        last_values (Optional[torch.Tensor]): Value estimates for final state (bootstrap) with shape (N, 1). This is required if the last state is not terminal.
        gamma (float): Discount factor

    Returns:
        torch.Tensor: The returns with shape that matches the input rewards shape.
    """
    if rewards.ndim != 3 or rewards.shape[-1] != 1:
        raise ValueError(f"`trajectory_returns` only supports shape (T, N, 1), but got {rewards.shape}")

    T, N, _ = rewards.shape
    returns = torch.zeros_like(rewards)

    if last_values is None:
        running_return = torch.zeros((N, 1), device=rewards.device)
    else:
        running_return = last_values

    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        returns[t] = running_return

    return returns

