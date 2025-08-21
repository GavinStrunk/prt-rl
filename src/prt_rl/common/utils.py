"""
Utility functions for reinforcement learning agents that are used across different algorithms.
"""
import random
import numpy as np
import torch
from typing import Tuple, Optional, Union, List

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


def clamp_actions(actions: torch.Tensor,
                  action_min: Union[float, List[float], torch.Tensor],
                  action_max: Union[float, List[float], torch.Tensor]) -> torch.Tensor:
    """
    Clamps actions to be within min and max bounds, handling scalar, list, or tensor bounds.

    Args:
        actions (torch.Tensor): Action tensor of shape (B, A)
        action_min (float | List[float] | torch.Tensor): Minimum bounds for actions.
        action_max (float | List[float] | torch.Tensor): Maximum bounds for actions.

    Returns:
        torch.Tensor: Clamped actions of shape (B, A)
    """
    if actions.dtype is not torch.float32:
        raise ValueError(f"Expected actions to be of type torch.float32, got {actions.dtype}")
    
    device = actions.device

    # Convert to tensors if needed
    if isinstance(action_min, (float, int)):
        action_min = torch.full((actions.shape[1],), float(action_min), device=device)
    elif isinstance(action_min, list):
        action_min = torch.tensor(action_min, device=device, dtype=actions.dtype)
    else:
        action_min = action_min.to(device).view(-1)

    if isinstance(action_max, (float, int)):
        action_max = torch.full((actions.shape[1],), float(action_max), device=device)
    elif isinstance(action_max, list):
        action_max = torch.tensor(action_max, device=device, dtype=actions.dtype)
    else:
        action_max = action_max.to(device).view(-1)

    # Reshape to (1, A) for broadcasting over batch
    action_min = action_min.view(1, -1)
    action_max = action_max.view(1, -1)

    return torch.clamp(actions, min=action_min, max=action_max)

def polyak_update(target: torch.nn.Module, network: torch.nn.Module, tau: float) -> None:
    """
    Updates a target network using Polyak averaging.

    When tau is 0 the target is unchanged and when tau is 1 a hard update is performed. The parameters of the target network are updated in place.

    .. math::
        \Theta_{target} = \tau * \Theta_{\pi} + (1 - \tau) * \Theta_{target}

    Args:
        target (torch.nn.Module): The target network to be updated.
        network (torch.nn.Module): The policy, pi, network from which parameters are taken.
        tau (float): The interpolation factor, typically in the range [0, 1].

    References:
    [1] https://github.com/DLR-RM/stable-baselines3/issues/93
    """
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

    .. math::
        A_{norm} = \frac{A - \mu}{\sigma + \epsilon}

    Args:
        advantages (torch.Tensor): The advantages to normalize. Shape (B, 1)

    Returns:
        torch.Tensor: The normalized advantages. Shape (B, 1)
    """
    # Handle the case where advantages is a single value. Mean is 0
    if advantages.numel() == 1:
        return torch.zeros_like(advantages)
    
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
        not_done = 1.0 - dones[t].float()
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
        last_values (Optional[torch.Tensor]): Value estimates for final state (bootstrap) with shape (N, 1) or (1, 1). This is required if the last state is not terminal or 0 is assumed for the last value.
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
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t].float())
        rtg[t] = running_return

    return rtg.view(original_shape)

def trajectory_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    last_values: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Computes the discounted returns for a sequence of rewards, also known as total discounted return. This function supports bootstrapping partial trajectories, as well as, flattened or time-major inputs.

    ..math ::
        \sum_{t'=0}^{T-1}\gamma^{t'}r(s_{i,t'},a_{i,t'}) + \gamma^{T}V(s_{i,T})

    When arguments are passed in with shape (B, 1) it is assumed these are stacked trajectories. The assumption is only the last trajectory is potentially not complete. 

    Args:
        rewards (torch.Tensor): Rewards from rollout with shape (T, N, 1) or (B, 1)
        dones (torch.Tensor): Done flags (1 if episode ended at step t, else 0) with shape (T, N, 1) or (B, 1)
        last_values (Optional[torch.Tensor]): Value estimates for final state (bootstrap) with shape (N, 1) or (1, 1). This is required if the last state is not terminal or 0 is assumed for the last value.
        gamma (float): Discount factor

    Returns:
        torch.Tensor: The returns with shape that matches the input rewards shape.
    """
    if rewards.shape != dones.shape:
        raise ValueError(f"`rewards` and `dones` must match shape. Got {rewards.shape} and {dones.shape}")

    # Save the original shape so we can reshape the output
    original_shape = rewards.shape

    # Case 1: time-major (T, N, 1)
    if rewards.ndim == 3:
        # Reshape rewards and dones to (T, N)
        T, N, _ = rewards.shape
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)

    # Case 2: flattened batch (B, 1)
    elif rewards.ndim == 2:
        # Treat as a single environment where T=B and N=1
        T, N = rewards.shape
        rewards = rewards.view(T, N)
        dones = dones.view(T, N)
    else:
        raise ValueError(f"Unsupported input shape: {rewards.shape}")

    G = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)
    discount = torch.ones(N, dtype=rewards.dtype, device=rewards.device)

    segment_returns = torch.zeros_like(rewards)
    segment_lengths = torch.zeros_like(rewards, dtype=torch.int)

    t_start = torch.zeros(N, dtype=torch.long, device=rewards.device)

    for t in range(T):
        G += rewards[t] * discount
        segment_lengths[t] = t - t_start + 1
        segment_returns[t] = G
        discount *= gamma

        # Check if any trajectory has ended and handle multiple trajectories
        done = dones[t] == 1.0
        if done.any():
            for i in range(N):
                # If a trajectory has a done copy the trajectory return for the entire segment and reset the returns to 0
                if done[i]:
                    segment_returns[t_start[i]:t + 1, i] = G[i]
                    t_start[i] = t + 1
                    G[i] = 0.0
                    discount[i] = 1.0

    # Bootstrap remaining segments if last_values are provided
    if last_values is not None:
        last_values = last_values.view(-1)
        for i in range(N):
            # If t_start is less than T, then there was not a done for the last value
            if t_start[i] < T:
                # Bootstrap by adding the discounted last value
                G[i] += (discount[i] * last_values[i])
                segment_returns[t_start[i]:, i] = G[i]

    return segment_returns.unsqueeze(-1).view(original_shape)

def gaussian_noise(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Tuple[int, ...] = (1,),
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generates Gaussian noise with specified mean and standard deviation.

    Args:
        mean (float): Mean of the Gaussian distribution.
        std (float): Standard deviation of the Gaussian distribution.
        shape (Tuple[int, ...]): Shape of the output tensor.

    Returns:
        torch.Tensor: A tensor filled with Gaussian noise.
    """
    return torch.normal(mean=mean, std=std, size=shape, dtype=torch.float32, device=device)

def ornstein_uhlenbeck_noise(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Tuple[int, ...] = (1,),
    theta: float = 0.15,
    dt: float = 1e-2,
    x0: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Generates Ornstein-Uhlenbeck noise for exploration in continuous action spaces.

    Orstein-Uhlenbeck noise is a stochastic process that modelled the velocity of a massive Brownian particle under the influence of friction. It is defined by the following stochastic differential equation:

    .. math::
        dx_t = \theta (\mu - x_t) dt + \sigma dW_t

    where :math:`\mu` is the mean, :math:`\sigma` is the standard deviation, and :math:`dW_t` is a Wiener process.

    This implementation uses the Euler-Maruyama method to discretize and approximate the process following the equation:

    .. math::
        x_{t+1} = x_t + \theta (\mu - x_t) dt + \sigma \delta W_t

    where :math:`\delta W_t \sim \mathcal{N}(0, \delta t) = \sqrt{\delta t}\mathcal{N}(0, 1)`.

    Args:
        mean (float): Mean of the noise.
        std (float): Standard deviation of the noise.
        shape (Tuple[int, ...]): Shape of the output tensor. Supports (B, action_dim)
        theta (float): Rate of mean reversion.
        dt (float): Time step size.
        x0 (Optional[torch.Tensor]): Initial value or previous value for the noise process. (if None, it will be initialized to zeros)

    Returns:
        torch.Tensor: A tensor filled with Ornstein-Uhlenbeck noise.

    References:
        [1] http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        [2] https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        [3] https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
    """
    if x0 is None:
        x0 = torch.zeros(shape)
    
    # Convert dt to tensor to make torch compatible
    dt = torch.tensor(dt, dtype=x0.dtype, device=x0.device)
    
    noise = x0 + theta * (mean - x0) * dt + std * torch.sqrt(dt) * torch.randn_like(x0)
    
    return noise
