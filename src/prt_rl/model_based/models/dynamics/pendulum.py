"""Pendulum dynamics model for physics-based simulation."""

import torch


class PendulumDynamics:
    """
    Physics-based dynamics model for a simple pendulum system.
    
    This class implements the equations of motion for a simple pendulum with
    torque control. The dynamics are based on the physical parameters of the
    pendulum (mass, length, gravity) and use Euler integration for time stepping.

    Note: Assumes zero theta corresponds to the pendulum straight up, and positive theta / torque is counterclockwise.
    
    Attributes:
        g (float): Gravitational acceleration constant (m/s²). Defaults to 10.0.
        l (float): Length of the pendulum (m).
        m (float): Mass of the pendulum bob (kg).
        dt (float): Time step for Euler integration (s).
    """
    
    def __init__(
        self, 
        g: float = 10.0, 
        l: float = 1.0, 
        m: float = 1.0,
        dt: float = 0.05
    ) -> None:
        """
        Initialize the pendulum dynamics model.
        
        Args:
            g: Gravitational acceleration constant in m/s². Defaults to 10.0.
            l: Length of the pendulum in meters. Defaults to 1.0.
            m: Mass of the pendulum bob in kg. Defaults to 1.0.
            dt: Time step for Euler integration in seconds. Defaults to 0.05.
        """
        self.g = g
        self.l = l
        self.m = m
        self.dt = dt

    def step(
        self, 
        theta: torch.Tensor, 
        thetad: torch.Tensor, 
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one time step of the pendulum dynamics using Euler integration.
        
        Args:
            theta: Pendulum angle in radians. Shape: [B, 1] where B is batch size.
            thetad: Pendulum angular velocity in rad/s. Shape: [B, 1].
            tau: Applied torque in N⋅m. Shape: [B, 1].
        
        Returns:
            Next state [theta, thetad] after one time step. Shape: [B, 2].
        """
        xdot = self.derivative(theta, thetad, tau)  # [B, 2]
        x = torch.cat([theta, thetad], dim=-1)
        x_next = x + xdot * self.dt
        return x_next
    
    def derivative(
        self, 
        theta: torch.Tensor, 
        thetad: torch.Tensor, 
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the time derivatives of the pendulum state.
        
        Uses the equations of motion for a simple pendulum with torque input:
        θ̈ = (3g)/(2l) * sin(θ) + 3/(ml²) * τ
        
        Args:
            theta: Pendulum angle in radians. Shape: [B, 1] where B is batch size.
            thetad: Pendulum angular velocity in rad/s. Shape: [B, 1].
            tau: Applied torque in N⋅m. Shape: [B, 1].
        
        Returns:
            Time derivatives [θ̇, θ̈]. Shape: [B, 2].
        """
        thetadd = (3 * self.g / (2 * self.l) * torch.sin(theta) + 3 / (self.m * self.l**2) * tau)
        return thetadd