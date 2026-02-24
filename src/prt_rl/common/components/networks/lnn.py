import torch
from torch import nn, Tensor
from typing import Tuple

class LagrangianNetwork(nn.Module):
  """
  Neural network for modeling the Lagrangian and generalized forces of a dynamical system.

  Args:
    state_dim (int): Dimension of the state (q, dq) input.
    hidden_dim (int, optional): Number of hidden units in each layer. Default is 64.
    action_dim (int | None, optional): Dimension of the action input. If provided, enables the Q_net for force computation.
  """
  def __init__(
    self,
    state_dim: int,
    hidden_dim: int = 64,
    action_dim: int | None = None
  ):
    super().__init__()
    self.L_net = nn.Sequential(
      nn.Linear(state_dim, hidden_dim),
      nn.Tanh(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.Tanh(),
      nn.Linear(hidden_dim, 1)  # Output is the scalar Lagrangian
    )
    self.action_dim = action_dim
    if action_dim is not None:
      self.Q_net = nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, action_dim)  # Output is the Q-value(s)
      )

  def lagrangian(self, q: Tensor, dq: Tensor) -> Tensor:
    """
    Compute the Lagrangian for given state and velocity.

    Args:
      q (Tensor): Generalized coordinates, shape (B, q_dim)
      dq (Tensor): Generalized velocities, shape (B, dq_dim)

    Returns:
      Tensor: Scalar Lagrangian values, shape (B, 1)
    """
    q = q.requires_grad_(True)
    dq = dq.requires_grad_(True)
    L = self.L_net(torch.cat([q, dq], dim=-1))  # [B, 1]
    return L

  def forces(self, q: Tensor, dq: Tensor, a: Tensor) -> Tensor:
    """
    Compute the generalized forces for given state, velocity, and action.

    Args:
      q (Tensor): Generalized coordinates, shape (B, q_dim)
      dq (Tensor): Generalized velocities, shape (B, dq_dim)
      a (Tensor): Actions, shape (B, action_dim)

    Returns:
      Tensor: Generalized forces, shape (B, action_dim)
    """
    if self.action_dim is None:
      raise ValueError("Action dimension must be specified to compute forces.")
    
    q = q.requires_grad_(True)
    dq = dq.requires_grad_(True)
    a = a.requires_grad_(True)
    Q = self.Q_net(torch.cat([q, dq, a], dim=-1))  # [B, action_dim]
    return Q
  
  def compute_gradients(self, q: Tensor, qd: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute the gradients of the Lagrangian with respect to q and dq.

    Args:
      q (Tensor): Generalized coordinates, shape (B, q_dim)
      qd (Tensor): Generalized velocities, shape (B, dq_dim)
    Returns:
      Tuple[Tensor, Tensor]:
        - dL_dq: Gradient of Lagrangian w.r.t. q, shape (B, q_dim)
        - dL_ddq: Gradient of Lagrangian w.r.t. dq, shape (B, dq_dim)
    """
    q = q.requires_grad_(True)
    qd = qd.requires_grad_(True)
    L = self.lagrangian(q, qd)  # [B, 1]

    dL_dq = torch.autograd.grad(
      outputs=L.sum(),
      inputs=q,
      create_graph=True,
    )[0]  # [B, q_dim]
    dL_ddq = torch.autograd.grad(
      outputs=L.sum(),
      inputs=qd,
      create_graph=True,
    )[0]  # [B, dq_dim]
    return dL_dq, dL_ddq
  
  def compute_ddq(
    self,
    dL_dq: Tensor,
    dL_dqd: Tensor,
    q: Tensor,
    qd: Tensor,
    tau: Tensor | None = None,
    eps: float = 1e-6
  ) -> Tensor:
    """
    Compute the acceleration (ddq) using the Euler-Lagrange equation.

    Args:
      dL_dq (Tensor): Gradient of Lagrangian w.r.t. q, shape (B, q_dim)
      dL_dqd (Tensor): Gradient of Lagrangian w.r.t. dq, shape (B, dq_dim)
      q (Tensor): Generalized coordinates, shape (B, q_dim)
      qd (Tensor): Generalized velocities, shape (B, dq_dim)
      tau (Tensor | None, optional): Generalized forces/torques, shape (B, q_dim). If None, zeros are used.
      eps (float, optional): Small value for numerical stability. Default is 1e-6.
    Returns:
      Tensor: Generalized accelerations (ddq), shape (B, q_dim)
    """
    M = torch.autograd.grad(dL_dqd.sum(), qd, create_graph=True)[0]  # [B, q_dim]
    M = torch.clamp(M, min=1e-3)
    C = torch.autograd.grad(dL_dqd.sum(), q, create_graph=True)[0]  # [B, q_dim]

    if tau is None:
      tau = torch.zeros_like(q)  # [B, q_dim]

    ddq = (dL_dq + tau - C * qd) / (M + eps)  # [B, q_dim]
    return ddq

  # def derivative(self, q: Tensor, qd: Tensor, tau: Tensor, eps: float = 1e-6) -> Tensor:
  #   dL_dq, dL_dqd = self.compute_gradients(q, qd)

  #   M = torch.autograd.grad(dL_dqd.sum(), qd, create_graph=True)[0] # [B, state_dim]
  #   M = torch.clamp(M, min=1e-3)
  #   C = torch.autograd.grad(dL_dqd.sum(), q, create_graph=True)[0] # [B, state_dim]

  #   if tau is None:
  #     tau = torch.zeros_like(q)  # [B, state_dim]

  #   ddq = (dL_dq + tau - C * qd) / (M + eps)  # [B, state_dim]
  #   return ddq 