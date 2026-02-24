import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple

class RandomPendulum(gym.Wrapper):
    """
    Domain randomization wrapper for Gymnasium Pendulum-v1.

    Randomizes episode-level physical parameters on every reset:
      - mass m : Default 1.0 kg
      - length l : Default 1.0 m
      - gravity g : Default 10.0 m/s^2

    Notes:
      - This wrapper assumes the underlying env exposes attributes: m, l, g (Gymnasium Pendulum does).
      - Parameters are sampled at reset() and stored in self.domain_params.
      - By default, domain params are returned only in reset info (not every step).

    Args:
        env: A Pendulum-v1 environment instance.
        m_scale: Relative scaling range applied to default mass (e.g., (0.8, 1.2)).
        l_scale: Relative scaling range applied to default length.
        g_scale: Relative scaling range applied to default gravity.
        m_abs: Absolute range for mass; if provided, overrides m_scale.
        l_abs: Absolute range for length; if provided, overrides l_scale.
        g_abs: Absolute range for gravity; if provided, overrides g_scale.
        include_in_step_info: If True, also include domain_params in step() info dict.
        randomization_seed: Seed for the domain-randomization RNG sequence.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        m_scale: Tuple[float, float] = (0.8, 1.2),
        l_scale: Tuple[float, float] = (0.8, 1.2),
        g_scale: Tuple[float, float] = (0.9, 1.1),
        m_abs: Optional[Tuple[float, float]] = None,
        l_abs: Optional[Tuple[float, float]] = None,
        g_abs: Optional[Tuple[float, float]] = None,
        include_in_step_info: bool = False,
        randomization_seed: Optional[int] = None,
    ):
        super().__init__(env)

        u = self.env.unwrapped
        # Cache the "nominal" defaults so scale ranges are stable across resets.
        # (Pendulum defaults are typically m=1.0, l=1.0, g=10.0, but we don’t hardcode.)
        self._nominal_m = float(getattr(u, "m"))
        self._nominal_l = float(getattr(u, "l"))
        self._nominal_g = float(getattr(u, "g"))

        self.m_scale = m_scale
        self.l_scale = l_scale
        self.g_scale = g_scale
        self.m_abs = m_abs
        self.l_abs = l_abs
        self.g_abs = g_abs
        self.include_in_step_info = include_in_step_info
        self.randomization_seed = randomization_seed
        self._dr_rng = np.random.default_rng(randomization_seed)

        self.domain_params: Dict[str, float] = {
            "m": self._nominal_m,
            "l": self._nominal_l,
            "g": self._nominal_g,
        }

    def _sample_param(self, *, nominal: float, scale: Tuple[float, float], abs_range: Optional[Tuple[float, float]]) -> float:
        if abs_range is not None:
            return float(self._dr_rng.uniform(abs_range[0], abs_range[1]))
        s = float(self._dr_rng.uniform(scale[0], scale[1]))
        return float(nominal * s)

    def _apply_domain_params(self) -> None:
        u = self.env.unwrapped

        m = self._sample_param(nominal=self._nominal_m, scale=self.m_scale, abs_range=self.m_abs)
        l = self._sample_param(nominal=self._nominal_l, scale=self.l_scale, abs_range=self.l_abs)
        g = self._sample_param(nominal=self._nominal_g, scale=self.g_scale, abs_range=self.g_abs)

        # Apply to underlying env (PendulumEnv uses these directly in dynamics)
        setattr(u, "m", m)
        setattr(u, "l", l)
        setattr(u, "g", g)

        self.domain_params = {"m": m, "l": l, "g": g}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset domain-randomization RNG when reset seed is provided.
        # This makes domain parameter sampling reproducible for identical reset seeds.
        if seed is not None:
            self._dr_rng = np.random.default_rng(seed)

        self._apply_domain_params()

        info = dict(info)
        info["domain_params"] = dict(self.domain_params)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.include_in_step_info:
            info = dict(info)
            info["domain_params"] = dict(self.domain_params)
        return obs, reward, terminated, truncated, info        

class POMDPPendulumWrapper(gym.Wrapper):
    """
    Wrapper that converts the Pendulum-v1 environment into a POMDP by only returning the angle (not angular velocity) in observations.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        # obs is [cos(theta), sin(theta), theta_dot]; we only return [cos(theta), sin(theta)]
        partial_obs = obs[:2]
        return partial_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        partial_obs = obs[:2]
        return partial_obs, reward, terminated, truncated, info