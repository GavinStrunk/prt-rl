from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import torch


# -------- tensor cache --------
# key: (id(config), device_str, dtype_str)  ->  value: dict[str, torch.Tensor]
_TENSOR_CFG_CACHE: Dict[Tuple[int, str, str], Dict[str, torch.Tensor]] = {}

def _tensorize_cfg_cached(config, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    key = (id(config), str(device), str(dtype))
    entry = _TENSOR_CFG_CACHE.get(key)
    if entry is None:
        d = asdict(config)
        entry = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in d.items()}
        _TENSOR_CFG_CACHE[key] = entry
    return entry

@dataclass
class CartPoleConfig:
    """
    Configuration parameters for the Inverted Pendulum model. The default parameters are modeled after the InvertedPendulum-v5 environment from Gymnasium.

    Attributes:
        M (float): Mass of the cart (kg).
        m (float): Mass of the pendulum (kg).
        l (float): Length of the pendulum (m).
        I (float): Moment of inertia of the pendulum (kg*m^2).
        b_cart (float): Coefficient of friction for the cart linear damping (Nm/s).
        b_pole(float): Coefficient of friction for the pendulum rotational damping (Nm/s).
        g (float): Acceleration due to gravity (m/s^2).
        dt (float): Time step for simulation (s).
        F_scale (float): Scaling factor for the applied force.
    """
    M: float = 10.472
    m: float = 5.019
    l: float = 1.0  
    I: float = 0.153
    b_cart: float = 1.0
    b_pole: float = 1.0
    g: float = 9.81  
    dt: float = 0.02  
    F_scale: float = 100.0

def cartpole_step(config: CartPoleConfig, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Compute the next state of the inverted pendulum given the current state and action using the equations of motion.

    Args:
        config (CartPoleConfig): Configuration parameters for the cart-pole system.
        state (torch.Tensor): Current state tensor of shape (batch_size, 4) where each state is [x, theta, x_dot, theta_dot].
        action (torch.Tensor): Action tensor of shape (batch_size, 1) representing the force applied to the cart.

    Returns:
        torch.Tensor: Next state tensor of shape (batch_size, 4).
    """
    device, dtype = state.device, state.dtype
    C = _tensorize_cfg_cached(config, device, dtype)

    x, theta, x_dot, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    F = action[:, 0] * C["F_scale"]

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    total_mass = C["M"] + C["m"]
    pole_mass_length = C["m"] * C["l"]

    # EOM (your form)
    temp = (F + pole_mass_length * theta_dot**2 * sin_theta - C["b_cart"] * x_dot) / total_mass
    theta_acc = (C["g"] * sin_theta - cos_theta * temp - C["b_pole"] * theta_dot / pole_mass_length) / \
                (C["l"] * (4.0/3.0 - C["m"] * cos_theta**2 / total_mass))
    x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

    x_next        = x        + x_dot      * C["dt"]
    theta_next    = theta    + theta_dot  * C["dt"]
    x_dot_next    = x_dot    + x_acc      * C["dt"]
    theta_dot_next= theta_dot+ theta_acc  * C["dt"]

    return torch.stack([x_next, theta_next, x_dot_next, theta_dot_next], dim=1)