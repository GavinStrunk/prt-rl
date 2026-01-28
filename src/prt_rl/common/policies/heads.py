import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Dict, Optional
import prt_rl.common.decision_functions as dfcn

class CategoricalHead(nn.Module):
    """
    Categorical actor head for discrete action spaces.

    Contract:
      - sample(latent, deterministic) -> (action, log_prob, entropy)
          action:   (B,)   int64
          log_prob: (B,1)  float
          entropy:  (B,1)  float
      - log_prob(latent, action) -> (B,1)
      - entropy(latent) -> (B,1)

    Notes:
      - Uses torch.distributions.Categorical(logits=...)
      - Deterministic action is argmax over logits.
    """    
    def __init__(self, latent_dim: int, num_actions: int) -> None:
        super().__init__()
        if num_actions <= 1:
            raise ValueError(f"n_actions must be > 1, got {num_actions}")
        self.n_actions = int(num_actions)
        self.logits = nn.Linear(latent_dim, num_actions)

    def _dist(self, latent: Tensor) -> torch.distributions.Categorical:
        logits = self.logits(latent)  # (B, n_actions)
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def sample(self, latent: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        dist = self._dist(latent)

        if deterministic:
            # dist.logits exists on Categorical; argmax on last dim yields (B,)
            action = dist.logits.argmax(dim=-1)
        else:
            action = dist.sample()  # (B,)

        log_prob = dist.log_prob(action).unsqueeze(-1)   # (B,1)
        entropy = dist.entropy().unsqueeze(-1)           # (B,1)
        return action, log_prob, entropy

    def log_prob(self, latent: Tensor, action: Tensor) -> Tensor:
        """
        action expected shape: (B,) (dtype long) or (B,1) which will be squeezed.
        """
        if action.ndim == 2 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        dist = self._dist(latent)
        return dist.log_prob(action).unsqueeze(-1)  # (B,1)

    def entropy(self, latent: Tensor) -> Tensor:
        dist = self._dist(latent)
        return dist.entropy().unsqueeze(-1)  # (B,1)
    
class ContinuousHead(nn.Module):
    """
    Continuous action head that outputs raw actions for continuous action spaces.

    This head is typically used in deterministic policy algorithms where the policy
    directly outputs continuous action values without sampling from a distribution.

    Notes:
      - The output layer is linear (no activation), producing raw action values.
      - The expected output shape is (B, action_dim), where B is the batch size
        and action_dim is the dimensionality of the action space.
    """

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.action_layer = nn.Linear(latent_dim, action_dim)

    def forward(self, latent: Tensor) -> Tensor:
        """
        Compute the continuous action output.

        Args:
            latent (Tensor): Latent state representation of shape (B, latent_dim).

        Returns:
            Tensor: Continuous action output of shape (B, action_dim).
        """
        return self.action_layer(latent)

class DecisionHead(nn.Module):
    """
    Decision head for discrete action spaces that outputs raw logits and applies a decision function.

    This head is typically used in deterministic policy algorithms where the policy
    directly outputs logits for discrete actions without sampling from a distribution.

    Notes:
      - The output layer is linear (no activation), producing raw logits.
      - The expected output shape is (B, action_dim), where B is the batch size
        and action_dim is the number of discrete actions.
    """

    def __init__(self, 
                 latent_dim: int, 
                 action_dim: int,
                 *,
                 decision_function: dfcn.DecisionFunction = dfcn.Greedy()
                 ) -> None:
        super().__init__()
        self.qval_layer = nn.Linear(latent_dim, action_dim)
        self.decision_function = decision_function

    def sample(self, latent: Tensor, deterministic: bool = False) -> Tensor:
        """
        Compute the decision logits output.

        Args:
            latent (Tensor): Latent state representation of shape (B, latent_dim).

        Returns:
            Tensor: Actions of shape (B,).
        """
        qvals = self.qval_layer(latent)

        if deterministic:
            # Deterministic action is argmax over logits.
            actions = qvals.argmax(dim=-1, keepdim=True)
        else:
            actions = self.decision_function.select_action(qvals)

        return actions

class GaussianHead(nn.Module):
    """
    Diagonal Gaussian actor head for continuous action spaces.

    Parameterization:
      - mean = Linear(latent)
      - log_std is a learned parameter vector (state-independent)

    Contract:
      - sample(latent, deterministic) -> (action, log_prob, entropy)
          action:   (B, act_dim)
          log_prob: (B,1)  summed over action dims
          entropy:  (B,1)  summed over action dims
      - log_prob(latent, action) -> (B,1)
      - entropy(latent) -> (B,1)

    Notes:
      - Uses torch.distributions.Normal(mean, std) with diagonal independence.
      - We sum log_prob/entropy over action dims inside the head to keep callers DRY.
      - If you want state-dependent std, replace the Parameter with another Linear head.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        log_std_init: float = -0.5,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        use_rsample: bool = False,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")

        self.action_dim = int(action_dim)
        self.mean = nn.Linear(latent_dim, action_dim)

        # state-independent log_std
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.use_rsample = bool(use_rsample)

    def _dist(self, latent: Tensor) -> torch.distributions.Normal:
        mean = self.mean(latent)  # (B, act_dim)
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def sample(self, latent: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        dist = self._dist(latent)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample() if self.use_rsample else dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # (B,1)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)          # (B,1)
        return action, log_prob, entropy

    def log_prob(self, latent: Tensor, action: Tensor) -> Tensor:
        """
        action expected shape: (B, act_dim)
        """
        dist = self._dist(latent)
        return dist.log_prob(action).sum(dim=-1, keepdim=True)  # (B,1)

    def entropy(self, latent: Tensor) -> Tensor:
        dist = self._dist(latent)
        return dist.entropy().sum(dim=-1, keepdim=True)  # (B,1)


class TanhGaussianHead(nn.Module):
    """
    Squashed (tanh) diagonal Gaussian actor head, commonly used in SAC.

    It parameterizes a Normal distribution in R^act_dim, samples with rsample(),
    then squashes via tanh to (-1, 1). Optionally scales/shifts to env bounds.

    Key detail: log_prob must include the tanh "change of variables" correction:
        log pi(a) = log N(u; mu, std) - sum log(1 - tanh(u)^2)
        where a = tanh(u)

    Optional scaling to [low, high]:
        a_env = scale * a + bias, scale=(high-low)/2, bias=(high+low)/2
      adds another correction term:
        log pi(a_env) = log pi(a) - sum log(scale)

    API:
      - sample(latent, deterministic=False) -> (action, log_prob, info)
      - log_prob(latent, action) -> log_prob (B,1)   [action is final env-scaled action]
      - entropy(...) is not analytic after tanh; SAC typically uses -log_prob as entropy term proxy.

    Notes:
      - Uses state-dependent mean and log_std by default (two linear layers).
      - If you want state-independent log_std, replace log_std_layer with nn.Parameter.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        epsilon: float = 1e-6,
        action_low: Optional[Tensor] = None,
        action_high: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")

        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.epsilon = float(epsilon)

        self.mean_layer = nn.Linear(latent_dim, action_dim)
        self.log_std_layer = nn.Linear(latent_dim, action_dim)

        # Optional action rescaling to env bounds.
        # Register as buffers so .to(device) moves them, and they are saved in state_dict.
        if action_low is not None or action_high is not None:
            if action_low is None or action_high is None:
                raise ValueError("Provide both action_low and action_high or neither.")
            if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
                raise ValueError(
                    f"Expected action_low/high shape ({action_dim},), "
                    f"got {tuple(action_low.shape)} and {tuple(action_high.shape)}"
                )
            self.register_buffer("action_low", action_low.clone().detach())
            self.register_buffer("action_high", action_high.clone().detach())
        else:
            self.action_low = None
            self.action_high = None

    def _base_dist(self, latent: Tensor) -> torch.distributions.Normal:
        mean = self.mean_layer(latent)  # (B, act_dim)
        log_std = self.log_std_layer(latent)  # (B, act_dim)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)

    def _squash(self, u: Tensor) -> Tensor:
        return torch.tanh(u)

    def _unsquash(self, a: Tensor) -> Tensor:
        """
        Inverse tanh (atanh). Input a should be in (-1,1).
        We clamp for numerical safety.
        """
        a = torch.clamp(a, -1.0 + self.epsilon, 1.0 - self.epsilon)
        return 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh

    def _apply_action_bounds(self, a: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Maps a in (-1,1) to env bounds, if provided.
        Returns (a_env, log_abs_det_jacobian_scale) where the latter is (B,1) or None.
        """
        if self.action_low is None:
            return a, None

        # scale and bias are (act_dim,)
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        a_env = a * scale + bias

        # log|det d a_env / d a| = sum log(scale)
        # constant per-dimension; broadcast to batch.
        log_scale = torch.log(scale).sum().expand(a.shape[0], 1)  # (B,1)
        return a_env, log_scale

    def _remove_action_bounds(self, a_env: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Inverse of _apply_action_bounds. Maps env-scaled action back to (-1,1).
        Returns (a, log_scale) where log_scale matches the forward mapping.
        """
        if self.action_low is None:
            return a_env, None

        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        a = (a_env - bias) / scale
        log_scale = torch.log(scale).sum().expand(a_env.shape[0], 1)  # (B,1)
        return a, log_scale

    @torch.no_grad()
    def sample(
        self,
        latent: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Returns:
          action:   (B, act_dim)  final action (env-scaled if bounds provided)
          log_prob: (B, 1)        log pi(action)
          info: dict containing:
              - "pre_tanh": u (B, act_dim)
              - "tanh": a in (-1,1) (B, act_dim)
              - "mean": mean (B, act_dim)
              - "log_std": log_std (B, act_dim)
        """
        dist = self._base_dist(latent)
        mean = dist.loc
        std = dist.scale
        log_std = torch.log(std + 1e-12)

        if deterministic:
            u = mean
        else:
            u = dist.rsample()  # reparameterized sample

        a = self._squash(u)  # (-1,1)

        # Log prob in squashed space (before optional scaling)
        # log N(u) - sum log(1 - tanh(u)^2)
        log_prob_u = dist.log_prob(u).sum(dim=-1, keepdim=True)  # (B,1)
        # stable correction: log(1 - tanh(u)^2) = log(1 - a^2)
        correction = torch.log(1.0 - a.pow(2) + self.epsilon).sum(dim=-1, keepdim=True)  # (B,1)
        log_prob = log_prob_u - correction  # (B,1)

        # Optional env scaling correction
        action, log_scale = self._apply_action_bounds(a)
        if log_scale is not None:
            log_prob = log_prob - log_scale

        info = {"pre_tanh": u, "tanh": a, "mean": mean, "log_std": log_std}
        return action, log_prob, info

    def log_prob(self, latent: Tensor, action: Tensor) -> Tensor:
        """
        Computes log pi(action) for a given *final* action (env-scaled if bounds were provided).

        action:
          - if bounds are None: expected in (-1,1)
          - if bounds are provided: expected in [low, high]
        """
        # Map env-scaled action back to tanh-space a in (-1,1)
        a, log_scale = self._remove_action_bounds(action)

        # Inverse tanh to get u
        u = self._unsquash(a)

        dist = self._base_dist(latent)
        log_prob_u = dist.log_prob(u).sum(dim=-1, keepdim=True)  # (B,1)
        correction = torch.log(1.0 - a.pow(2) + self.epsilon).sum(dim=-1, keepdim=True)  # (B,1)
        log_prob = log_prob_u - correction

        if log_scale is not None:
            log_prob = log_prob - log_scale

        return log_prob

    def entropy_proxy(self, latent: Tensor, action: Tensor) -> Tensor:
        """
        SAC typically uses -log_prob as an entropy term (up to expectations).
        This returns (B,1) = -log pi(a).

        You might prefer calling this `neg_log_prob` depending on your style.
        """
        return -self.log_prob(latent, action)

class BetaHead(nn.Module):
    """
    Beta actor head using torch.distributions.Beta.

    Produces independent Beta(alpha_i, beta_i) per action dimension.
    Samples in (0,1) and optionally maps to [action_low, action_high].

    API:
      - sample(latent, deterministic=False) -> (action, log_prob, entropy)
      - log_prob(latent, action) -> (B,1)   [action is final env-scaled action if bounds provided]
      - entropy(latent) -> (B,1)

    Notes:
      - Beta is defined on (0,1). We clamp actions for numerical stability.
      - If bounds are provided, we apply a change-of-variables correction to log_prob.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        concentration_offset: float = 1.0,   # encourages alpha,beta >= 1 initially
        min_concentration: float = 1e-3,     # numerical floor
        epsilon: float = 1e-6,               # clamp for actions near {0,1}
        action_low: Optional[Tensor] = None,
        action_high: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        self.action_dim = int(action_dim)

        self.concentration_offset = float(concentration_offset)
        self.min_concentration = float(min_concentration)
        self.epsilon = float(epsilon)

        self.alpha_layer = nn.Linear(latent_dim, action_dim)
        self.beta_layer = nn.Linear(latent_dim, action_dim)
        self.softplus = nn.Softplus()

        # Optional bounds
        if action_low is not None or action_high is not None:
            if action_low is None or action_high is None:
                raise ValueError("Provide both action_low and action_high or neither.")
            if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
                raise ValueError(
                    f"Expected action_low/high shape ({action_dim},), "
                    f"got {tuple(action_low.shape)} and {tuple(action_high.shape)}"
                )
            self.register_buffer("action_low", action_low.clone().detach())
            self.register_buffer("action_high", action_high.clone().detach())
        else:
            self.action_low = None
            self.action_high = None

    def _concentrations(self, latent: Tensor) -> Tuple[Tensor, Tensor]:
        # softplus -> (0, inf). offset helps avoid extreme U-shapes early in training.
        alpha = self.softplus(self.alpha_layer(latent)) + self.concentration_offset
        beta = self.softplus(self.beta_layer(latent)) + self.concentration_offset

        # hard floor for numerical safety
        alpha = torch.clamp(alpha, min=self.min_concentration)
        beta = torch.clamp(beta, min=self.min_concentration)
        return alpha, beta

    def _dist(self, latent: Tensor) -> torch.distributions.Beta:
        alpha, beta = self._concentrations(latent)
        # torch.distributions.Beta uses concentration1 (alpha) and concentration0 (beta)
        return torch.distributions.Beta(concentration1=alpha, concentration0=beta)

    def _apply_bounds(self, a01: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Map from (0,1) to [low, high] if bounds provided.
        Returns (action_env, log_scale) where log_scale is (B,1) = sum log(scale).
        """
        if self.action_low is None:
            return a01, None
        scale = (self.action_high - self.action_low)  # (act_dim,)
        action = self.action_low + a01 * scale
        log_scale = torch.log(scale).sum().expand(a01.shape[0], 1)  # (B,1)
        return action, log_scale

    def _remove_bounds(self, action_env: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Inverse of _apply_bounds.
        Returns (a01, log_scale) where log_scale matches forward mapping.
        """
        if self.action_low is None:
            return action_env, None
        scale = (self.action_high - self.action_low)
        a01 = (action_env - self.action_low) / scale
        log_scale = torch.log(scale).sum().expand(action_env.shape[0], 1)
        return a01, log_scale

    @torch.no_grad()
    def sample(self, latent: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        dist = self._dist(latent)

        if deterministic:
            # Mean of Beta(alpha,beta) = alpha/(alpha+beta)
            alpha = dist.concentration1
            beta = dist.concentration0
            a01 = alpha / (alpha + beta)
        else:
            a01 = dist.sample()

        # Keep away from boundaries for stable log_prob
        a01 = torch.clamp(a01, self.epsilon, 1.0 - self.epsilon)

        log_prob_01 = dist.log_prob(a01).sum(dim=-1, keepdim=True)  # (B,1)
        entropy_01 = dist.entropy().sum(dim=-1, keepdim=True)       # (B,1)

        action, log_scale = self._apply_bounds(a01)
        log_prob = log_prob_01 - log_scale if log_scale is not None else log_prob_01
        return action, log_prob, entropy_01

    def log_prob(self, latent: Tensor, action: Tensor) -> Tensor:
        a01, log_scale = self._remove_bounds(action)
        a01 = torch.clamp(a01, self.epsilon, 1.0 - self.epsilon)
        dist = self._dist(latent)
        log_prob_01 = dist.log_prob(a01).sum(dim=-1, keepdim=True)
        return log_prob_01 - log_scale if log_scale is not None else log_prob_01

    def entropy(self, latent: Tensor) -> Tensor:
        dist = self._dist(latent)
        return dist.entropy().sum(dim=-1, keepdim=True)

class ValueHead(nn.Module):
    """
    State-value function head.

    Implements a scalar value function V(s) that maps a latent state representation
    (typically produced by an encoder and/or backbone network) to a single value
    estimate per state.

    This head is commonly used by actor–critic algorithms such as PPO, A2C, and A3C.

    Notes:
        - The output layer is linear (no activation), as value functions are
          generally unbounded.
        - The expected output shape is (B, 1), where B is the batch size.

    Args:
        latent_dim (int): Dimension of the latent state representation.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.v = nn.Linear(latent_dim, 1)

    def forward(self, latent: Tensor) -> Tensor:
        """
        Compute the state-value estimate.

        Args:
            latent (Tensor): Latent state representation of shape (B, latent_dim).

        Returns:
            Tensor: Value estimates of shape (B, 1).
        """
        return self.v(latent)

    
class QValueHead(nn.Module):
    """
    State–action value function head.

    Implements a scalar Q-function Q(s, a) that maps a latent state representation
    and a continuous action vector to a single value estimate per state–action pair.

    This head is typically used by off-policy algorithms such as SAC and TD3,
    where the critic evaluates specific actions rather than all actions at once.

    Notes:
        - The latent state and action are concatenated along the feature dimension.
        - The output layer is linear (no activation), as Q-values are unbounded.
        - This implementation assumes continuous action spaces. For discrete
          action spaces (e.g., DQN), a different head that outputs Q(s, ·) is used.

    Args:
        latent_dim (int): Dimension of the latent state representation.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.q = nn.Linear(latent_dim, 1)

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        """
        Compute the state–action value estimate.

        Args:
            latent (Tensor): Latent state representation of shape (B, latent_dim).
            action (Tensor): Action tensor of shape (B, action_dim).

        Returns:
            Tensor: Q-value estimates of shape (B, 1).
        """
        q_input = torch.cat([latent, action], dim=1)
        return self.q(q_input)

class DuelingHead(nn.Module):
    """
    Dueling network head for discrete action spaces.

    Implements the dueling architecture where separate streams estimate
    the state-value function V(s) and the advantage function A(s, a).
    The Q-values are computed as:
        Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')
    This head is commonly used in DQN variants to improve value estimation.

    Args:
        latent_dim (int): Dimension of the latent state representation.
        num_actions (int): Number of discrete actions.
    """
    
    def __init__(self, latent_dim: int, num_actions: int):
        super().__init__()
        if num_actions <= 1:
            raise ValueError(f"num_actions must be > 1, got {num_actions}")
        self.num_actions = int(num_actions)
        self.value_stream = nn.Linear(latent_dim, 1)
        self.advantage_stream = nn.Linear(latent_dim, num_actions)

    def forward(self, latent: Tensor) -> Tensor:
        """
        Compute the Q-value estimates for all actions.

        Args:
            latent (Tensor): Latent state representation of shape (B, latent_dim). 
        Returns:
            Tensor: Q-value estimates of shape (B, num_actions).
        """
        V = self.value_stream(latent)               # (B, 1)
        A = self.advantage_stream(latent)           # (B, num_actions)
        A_mean = A.mean(dim=1, keepdim=True)       # (B, 1)
        Q = V + A - A_mean                          # (B, num_actions)
        return Q