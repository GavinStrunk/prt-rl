from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


Tensor = torch.Tensor


@dataclass
class LNNConfig:
    """Configuration for a Lagrangian Neural Network."""
    q_dim: int
    u_dim: int = 0                     # set >0 for action-conditioned / forced dynamics
    lagrangian_hidden: Tuple[int, ...] = (128, 128)
    force_hidden: Tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    learn_forces: bool = True          # if False, assumes Q = 0 (unforced)
    damping: bool = False              # optional dissipative term
    gravity: bool = False              # optional learned potential structure
    eps: float = 1e-6                  # numerical stability
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None


class LagrangianNN(nn.Module):
    """
    Lagrangian Neural Network with optional action-conditioned generalized forces.

    - Unforced (conservative): u_dim=0 or learn_forces=False => Q ≡ 0
    - Forced/action-conditioned: u_dim>0 and learn_forces=True => Q = Q(q, dq, u)

    Intended usage:
      L = model.lagrangian(q, dq)
      Q = model.generalized_forces(q, dq, u)  # optional
      ddq = model.accelerations(q, dq, u)     # used in loss
    """

    def __init__(self, cfg: LNNConfig):
        super().__init__()
        ...

    # ----------------------------
    # Core API
    # ----------------------------
    def lagrangian(self, q: Tensor, dq: Tensor) -> Tensor:
        """
        Compute scalar Lagrangian L(q, dq) per batch element.

        Args:
            q:  [B, n]
            dq: [B, n]
        Returns:
            L:  [B, 1] (or [B])
        """
        ...

    def generalized_forces(self, q: Tensor, dq: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """
        Compute generalized external forces Q.

        Args:
            q:  [B, n]
            dq: [B, n]
            u:  [B, m] (optional; required if action-conditioned)
        Returns:
            Q:  [B, n]
        """
        ...

    def accelerations(self, q: Tensor, dq: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """
        Compute accelerations ddq from Euler–Lagrange with forcing:
            d/dt(∂L/∂dq) - ∂L/∂q = Q

        Args:
            q:  [B, n]
            dq: [B, n]
            u:  [B, m] (optional)
        Returns:
            ddq: [B, n]
        """
        ...

    # ----------------------------
    # Autodiff helpers (useful for debugging / custom losses)
    # ----------------------------
    def dL_dq(self, q: Tensor, dq: Tensor) -> Tensor:
        """Return ∂L/∂q, shape [B, n]."""
        ...

    def dL_ddq(self, q: Tensor, dq: Tensor) -> Tensor:
        """Return ∂L/∂dq, shape [B, n]."""
        ...

    def mass_matrix(self, q: Tensor, dq: Tensor) -> Tensor:
        """
        Return M(q, dq) = ∂²L/∂dq², shape [B, n, n].
        Often used as a learned inertia / metric term.
        """
        ...

    def coriolis_and_gravity(self, q: Tensor, dq: Tensor) -> Tensor:
        """
        Return a convenience term C(q,dq) typically representing:
            C(q,dq) = ∂/∂q(∂L/∂dq) dq - ∂L/∂q
        (Exact definition may vary by implementation.)
        Shape [B, n].
        """
        ...

    def energy(self, q: Tensor, dq: Tensor) -> Tensor:
        """
        Return total energy E(q,dq) = dqᵀ(∂L/∂dq) - L.
        Shape [B, 1] (or [B]).
        """
        ...


    # ----------------------------
    # Validation / shape guards
    # ----------------------------
    def is_action_conditioned(self) -> bool:
        """Return True if model expects actions (u_dim > 0 and learn_forces)."""
        ...

