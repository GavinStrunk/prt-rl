import torch
from torch.distributions import Normal
from typing import Any, Callable, Optional
from prt_rl.model_based.planners.rollout import rollout_action_sequence

def temporal_smooth(
    x: torch.Tensor,
    method: str = "none",
    rho: float = 0.9,
    kernel_size: int = 0,
) -> torch.Tensor:
    """
    Apply simple temporal smoothing along the horizon dimension.

    Args:
        x : torch.Tensor
            Tensor of shape (N, H, dA) where:
            - N: number of sequences/samples
            - H: planning horizon
            - dA: action dimension
        method : {"none", "ou", "conv"}, default: "none"
            - "none": return x unchanged.
            - "ou":   Exponential moving average (Ornstein–Uhlenbeck-like) smoothing:
                    out[:, t] = rho * out[:, t-1] + (1 - rho) * x[:, t]
            - "conv": 1D convolution with a Gaussian-ish kernel of length `kernel_size`.
        rho : float, default: 0.9
            Smoothing factor for "ou". Higher means smoother (more inertia).
        kernel_size : int, default: 0
            Kernel length for "conv". Must be >= 3 to have an effect.

    Returns:
        torch.Tensor
            Smoothed tensor with the same shape as `x` (N, H, dA).

    Notes:
        - Smoothing should be applied in **U-space** for tanh bound mode (preferred),
        and in **A-space** for clip mode.
        - For "conv", edges are handled with 'replicate' padding.
    """
    N, H, da = x.shape

    if method == 'ou':
        smooth_x = x.clone()
        for t in range(1, H):
            smooth_x[:, t] = rho * smooth_x[:, t-1] + (1 - rho) * x[:, t]
        return smooth_x
    elif method == 'conv':
        t = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (t / (0.25 * kernel_size))**2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)  # (1,1,k)
        xt = x.permute(0, 2, 1).reshape(N * da, 1, H)  # (N*dA,1,H)
        pad = (kernel_size // 2, kernel_size // 2)
        xt = torch.nn.functional.pad(xt, pad, mode='replicate')
        yt = torch.nn.functional.conv1d(xt, kernel)  # (N*dA,1,H)
        return yt.view(N, da, H).permute(0, 2, 1).contiguous()
    else:
        return x

class CrossEntropyMethodPlanner:
    """
    Cross-Entropy Method (CEM) planner for continuous control with support for
    tanh-squash (U-space) and clip (A-space) bounding strategies.

    Workflow per planning call
    --------------------------
    1) Initialize or warm-start the sequence distribution (shape (H, dA)).
    2) Repeat for K iterations:
       a) Sample N sequences (N,H,dA) using the bound strategy.
       b) Roll out through the (known or learned) dynamics model.
       c) Compute reward for each sequence, pick top M (elites).
       d) Refit the distribution from elites (in the proper space).
    3) Return the first action of the best-scoring elite.

    Parameters
    ----------
    action_mins, action_maxs : torch.Tensor
        Action bounds with shape (dA, 1) (or (dA,)). Broadcasted internally.
    num_action_sequences : int, default: 100
        N, number of sequences sampled per iteration.
    planning_horizon : int, default: 10
        H, number of steps in each sequence.
    num_elites : int, default: 10
        M, number of top sequences used for refit.
    num_iterations : int, default: 5
        K, number of CEM refinement iterations per plan call.
    use_smoothing : bool, default: False
        If True, apply temporal smoothing (OU) inside the bound strategy
        (U-space for tanh, A-space for clip).
    use_clipping : bool, default: False
        If True, use ClipBound; otherwise use TanhSquashBound.
    tau : float or None, default: H/3
        Time constant for std decay schedule.
    beta : float, default: 0.2
        Long-horizon std floor fraction for the decay schedule.
    device : {"cpu","cuda",...}, default: "cpu"
        Device for internal tensors.

    Notes
    -----
    - This implementation assumes **higher reward is better**. If you use costs,
      either flip the sign or use `largest=False` in `topk`.
    - `rollout_action_sequence(model_config, model_fcn, state, actions)` must return
      a dict with `'state'`, `'action'`, and `'next_state'` batches consistent with
      shapes (N, H, ·).
    """
    def __init__(self,
                 action_mins: torch.Tensor,
                 action_maxs: torch.Tensor,                  
                 num_action_sequences: int = 100,
                 planning_horizon: int = 10,
                 num_elites: int = 10,
                 num_iterations: int = 5,
                 use_smoothing: bool = False,           
                 use_clipping: bool = False,
                 tau: float | None = None,
                 beta: float = 0.2,
                 device: str = 'cpu'
                 ) -> None:
        assert action_mins.shape == action_maxs.shape, "Action mins and maxs must have the same shape."
        assert num_elites <= num_action_sequences, "Number of elites must be less than or equal to number of action sequences."
        assert num_iterations > 0, "Number of iterations must be greater than 0."

        self.planning_horizon = planning_horizon
        self.num_action_sequences = num_action_sequences
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.use_smoothing = use_smoothing
        self.use_clipping = use_clipping
        self.tau = tau if tau is not None else planning_horizon / 3
        self.beta = beta
        self.device = torch.device(device)

        # Move action bound tensors to the correct device and compute the scale and bias for rescaling
        self.action_mins = action_mins.to(self.device)
        self.action_maxs = action_maxs.to(self.device)
        self.action_scale = (self.action_maxs - self.action_mins) / 2.0
        self.action_bias = (self.action_maxs + self.action_mins) / 2

        if self.use_clipping:
            self.bound_strategy = ClipBound()
        else:
            self.bound_strategy = TanhSquashBound()

        self.distribution = None
        self.elites = None

    def plan(self, 
             model_fcn: Callable, 
             model_config: Any, 
             reward_fcn: Callable, 
             state: torch.Tensor
             ) -> torch.Tensor:
        """
        Run one CEM planning call and return the first action to execute.

        Parameters
        ----------
        model_fcn : Callable
            One-step dynamics function (batched) used by the rollout utility.
        model_config : Any
            Additional config passed to your rollout helper.
        reward_fcn : Callable
            Function computing rewards from rollout dict; returns (N,) reward per sequence.
        state : torch.Tensor
            Current state (batching left to caller/rollout helper).

        Returns
        -------
        torch.Tensor
            First action of the best elite sequence, shape (1, dA).

        Notes
        -----
        - Sampling returns **A-space** actions in both strategies.
        - Refit is done in U or A space depending on the bound strategy.
        """
        # Initialize the prior distribution
        if self.distribution is None or self.elites is None:
           self.distribution = self.bound_strategy.cold_start(H=self.planning_horizon,
                                           a_mins=self.action_mins,
                                           a_maxs=self.action_maxs,
                                           beta=self.beta,
                                           tau=self.tau
                                           )
        else:
            self.distribution = self.bound_strategy.warm_start(elites=self.elites,
                                           a_mins=self.action_mins,
                                           a_maxs=self.action_maxs,
                                           widening_factor=1.3,
                                           std_min=0.5
                                           )        

        for _ in range(self.num_iterations):
            # Sample new action sequences - (N, H, da)
            action_sequences = self.bound_strategy.sample(self.distribution, torch.Size((self.num_action_sequences,)), self.action_mins, self.action_maxs)

            # Evaluate action sequences using the model and reward function
            rollout = rollout_action_sequence(model_config, model_fcn, state, action_sequences)
            rewards = reward_fcn(rollout['state'], rollout['action'], rollout['next_state']) 

            # Pick the top M elites
            _, elite_indices = torch.topk(rewards, self.num_elites, largest=True)
            self.elites = action_sequences[elite_indices]

            # Refit the distribution to the elites
            self.distribution = self.bound_strategy.refit(self.elites, self.action_mins, self.action_maxs)

        # Return the first action from the best action sequence
        return self.elites[0, 0, :].unsqueeze(0)


class TanhSquashBound:
    """
    Tanh squashing strategy (recommended default).

    The underlying distribution lives in **U-space** (unbounded). Sampling:
      U ~ Normal(mu_u, sigma_u)  (shape (H, dA))
      A = (tanh(U) + 1)/2 * (a_max - a_min) + a_min  (shape (N, H, dA), in bounds)

    Refit must be done in **U-space**. We therefore convert A-space elites back to
    U-space via an atanh-like transform and compute the new Normal in U.

    All methods here are stateless utilities; the planner owns the current distribution.

    Shapes
    ------
    - a_mins, a_maxs : (dA, 1) or (dA,)
    - distribution.loc/scale : (H, dA)
    - samples / elites : (N, H, dA)
    """   
    @staticmethod
    def sample(
        distribution: Normal,
        shape: torch.Size,
        a_mins: torch.Tensor,
        a_maxs: torch.Tensor,
        smoothing: str = "ou",
        rho: float = 0.5,
        kernel_size: int = 0,
    ) -> torch.Tensor:
        """
        Sample actions from a U-space Normal, optionally smooth in U, and squash to A-space.

        Parameters
        ----------
        distribution : Normal
            U-space Normal with loc/scale shape (H, dA).
        shape : torch.Size
            Leading sample shape, e.g., (N,) to get (N,H,dA).
        a_mins, a_maxs : torch.Tensor
            Bounds, shape (dA,1) or (dA,).
        smoothing : {"none","ou","conv"}, default: "none"
            Smoothing applied in **U-space** before squashing.
        rho : float, default: 0.9
            OU smoothing factor.
        kernel_size : int, default: 0
            Convolution kernel length (only used if smoothing="conv").

        Returns
        -------
        torch.Tensor
            Actions in A-space with shape (N, H, dA).
        """
        # Sample distribution in U-space with shape (N, H, da)
        u_actions = distribution.rsample(shape)

        # Apply temporal smoothing to the u-space actions
        u_smooth = temporal_smooth(u_actions, method=smoothing, rho=rho, kernel_size=kernel_size)

        # Convert action from U-space to action space
        a_actions = TanhSquashBound._from_u_space(u_smooth, a_mins, a_maxs)
        return a_actions
    
    @staticmethod
    def refit(
        elites: torch.Tensor,
        a_mins: torch.Tensor,
        a_maxs: torch.Tensor,
        std_min: float = 1e-6,
    ) -> Normal:
        """
        Refit the U-space Normal from A-space elites.

        Parameters
        ----------
        elites : torch.Tensor
            Elite action sequences in A-space, shape (N_e, H, dA).
        a_mins, a_maxs : torch.Tensor
            Bounds with shape (dA,1) or (dA,).
        std_min : float, default: 1e-6
            Std floor.

        Returns
        -------
        Normal
            Updated U-space Normal with loc/scale (H, dA).
        """
        # Convert elite actions to U-space
        u_elites = TanhSquashBound._to_u_space(elites, a_mins, a_maxs)
        mean_u = u_elites.mean(dim=0)
        std_u = u_elites.std(dim=0, unbiased=False).clamp_min(std_min)
        return Normal(loc=mean_u, scale=std_u)
    
    @staticmethod
    def cold_start(
        H: int,
        a_mins: torch.Tensor,
        a_maxs: torch.Tensor,
        beta: float,
        tau: float,
        sigma_u0: float = 0.6,
        std_min: float = 1e-6,
    ) -> Normal:
        """
        Initialize the U-space Normal used for tanh squashing.

        Parameters
        ----------
        H : int
            Planning horizon.
        a_mins, a_maxs : torch.Tensor
            Bounds with shape (dA,1) or (dA,). Used only for dtype/device;
            U-space init is centered at zero regardless of bounds.
        beta : float
            Long-horizon std decay floor in [0,1]. Effective std(t) = (beta + (1-beta) * exp(-t/tau)) * sigma_u0.
        tau : float
            Decay time constant (in steps).
        sigma_u0 : float, default: 0.6
            Initial U-space std per dimension at t=0 (before decay). Values in [0.4, 1.0] are robust.
        std_min : float, default: 1e-6
            Absolute std floor for numerical stability.

        Returns
        -------
        torch.distributions.Normal
            U-space Normal with loc/scale shape (H, dA).
        """
        # Get the action dimension
        da = a_mins.shape[0]

        # Compute the initial mean and standard deviation
        # Center of the action box
        center = ((a_mins + a_maxs) / 2.0).squeeze(-1)                  # (dA,)
        sigma_0 = ((a_maxs - a_mins) / 2.0).squeeze(-1)                 # (dA,)
        t = torch.arange(H, device=a_mins.device, dtype=center.dtype)   # (H,)
        decay = beta + (1-beta) * torch.exp(-t / tau)                   # (H,)

        # U-space Gaussian (pre-squash)
        # Choose a sigma_0 [0.4, 1.0] for robustness in U-space
        mean_u = torch.zeros(H, da, device=a_mins.device, dtype=center.dtype)   # (H, dA)
        sigma_0 = torch.full_like(center, sigma_u0)                                  # (dA,)
        std_u = decay.unsqueeze(1) * sigma_0.unsqueeze(0)                       # (H, dA)

        return Normal(
            loc=mean_u,
            scale=std_u.clamp_min(std_min)
        )
    
    @staticmethod
    def warm_start(
        elites: torch.Tensor,
        a_mins: torch.Tensor,
        a_maxs: torch.Tensor,
        widening_factor: float = 1.3,
        std_min: float = 1e-6,
    ) -> Normal:
        """
        Warm-start the U-space Normal from previous elites in A-space.

        Steps
        -----
        1) Convert elites A -> U, compute mean/std across elite batch.
        2) Shift μ_u, σ_u forward by one time-step.
        3) Tail: set last μ_u to 0 (center in U), keep last σ_u.
        4) Widen σ_u[0] by `widening_factor` to retain agility.
        5) (Optional) Anchor μ_u[0] toward the last executed action (converted to U)
           with convex blend μ_u[0] ← λ * μ_u[0] + (1-λ) * u_exec.

        Parameters
        ----------
        elites : (N_e, H, dA)
            Elite **A-space** action sequences from previous iteration.
        a_mins, a_maxs : (dA,1) or (dA,)
            Bounds.
        widening_factor : float, default: 1.3
            Multiplier for σ_u at t=0 after shift.
        std_min : float, default: 1e-6
            Std floor.
        executed_action : (1, dA) or (dA,), optional
            Last executed action to anchor to (A-space).
        anchor_lambda : float, default: 0.8
            Blend weight; larger = rely more on shifted mean.

        Returns
        -------
        Normal
            Warm-started U-space Normal (H, dA).
        """
        # Convert elite actions to U-space
        u_elites = TanhSquashBound._to_u_space(elites, a_mins, a_maxs)

        mean = torch.mean(u_elites, dim=0)
        standard_dev = torch.std(u_elites, dim=0, unbiased=False)

        # Shift the mean and std to the next time step
        shifted_mean = torch.zeros_like(mean, device=mean.device, dtype=mean.dtype)
        shifted_mean[:-1] = mean[1:]
        shifted_std = torch.zeros_like(standard_dev, device=standard_dev.device, dtype=standard_dev.dtype)
        shifted_std[:-1] = standard_dev[1:]

        # Add tail value for the last time step
        shifted_mean[-1].zero_()
        shifted_std[-1] = standard_dev[-1]

        # Widen the standard deviation to encourage exploration
        shifted_std[0] = (shifted_std[0] * widening_factor).clamp_min_(std_min)

        return Normal(
            loc=shifted_mean,
            scale=shifted_std.clamp_min(1e-6)
        )
    
    @staticmethod
    def _to_u_space(a_actions, a_mins, a_maxs, epsilon=1e-6):
        """
        Convert bounded actions A in [a_min, a_max] to U-space via atanh.

        Parameters
        ----------
        a : (N,H,dA)
        a_mins, a_maxs : (dA,1) or (dA,)
        eps : float
            Safety margin to avoid infinities at ±1 after normalization.

        Returns
        -------
        torch.Tensor
            U-space tensor (N,H,dA).
        """        
        y = (2*(a_actions - a_mins) / (a_maxs - a_mins) - 1).clamp(-1 + epsilon, 1 - epsilon)
        # atanh(y) = 0.5*log((1+y)/(1-y))
        return 0.5 * torch.log1p(y) - 0.5 * torch.log1p(-y)
    @staticmethod
    def _from_u_space(u: torch.Tensor, a_mins: torch.Tensor, a_maxs: torch.Tensor) -> torch.Tensor:
        """
        Map U-space to A-space via tanh and affine bounds.

        Parameters
        ----------
        u : (N,H,dA)
        a_mins, a_maxs : (dA,1) or (dA,)

        Returns
        -------
        torch.Tensor
            Bounded A-space actions (N,H,dA).
        """
        y = torch.tanh(u)
        return (y + 1) / 2 * (a_maxs - a_mins) + a_mins
      
class ClipBound:
    """
    Hard clipping strategy (simple & useful for bang-bang optima).

    The distribution lives and is refit in **A-space**. Sampling:
      A ~ Normal(mu_a, sigma_a)   (shape (H, dA))
      A := clamp(A, [a_min, a_max])

    Shapes
    ------
    - a_mins, a_maxs : (dA, 1) or (dA,)
    - distribution.loc/scale : (H, dA)
    - samples / elites : (N, H, dA)
    """    
    @staticmethod
    def sample(
        distribution: Normal,
        shape: torch.Size,
        a_mins: torch.Tensor,
        a_maxs: torch.Tensor,
        smoothing: str = "ou",
        rho: float = 0.5,
        kernel_size: int = 0,
    ) -> torch.Tensor:
        """
        Sample actions from an A-space Normal, optionally smooth in A, then clamp to bounds.

        Returns
        -------
        torch.Tensor
            A-space actions with shape (N, H, dA), in-bounds if `clamp=True`.
        """
        # Sample distribution in A-space with shape (N, H, da)
        a_actions = distribution.rsample(shape)

        # Apply temporal smoothing to the actions
        a_smooth = temporal_smooth(a_actions, method=smoothing, rho=rho, kernel_size=kernel_size)

        # Clip to the action bounds
        actions = torch.clamp(a_smooth, a_mins, a_maxs)
        return actions
    
    @staticmethod
    def refit(elites: torch.Tensor, a_mins: torch.Tensor, a_maxs: torch.Tensor, std_min: float = 1e-6) -> Normal:
        mean = elites.mean(dim=0)
        standard_dev = elites.std(dim=0, unbiased=False).clamp_min(std_min)
        return Normal(loc=mean, scale=standard_dev)
    
    @staticmethod
    def cold_start(
        H: int,
        a_mins: torch.Tensor,
        a_maxs: torch.Tensor,
        beta: float,
        tau: float,
        std_min: float = 1e-6,
    ) -> Normal:
        """
        Initialize an A-space Normal with center-of-box mean and decayed half-span std.

        sigma_a(t) = (beta + (1 - beta) * exp(-t/tau)) * (a_max - a_min)/2

        Returns
        -------
        Normal
            A-space Normal (H, dA).
        """
        # Get the action dimension
        da = a_mins.shape[0]

        # Compute the initial mean and standard deviation
        # Center of the action box
        center = ((a_mins + a_maxs) / 2.0).squeeze(-1)                  # (dA,)
        sigma_0 = ((a_maxs - a_mins) / 2.0).squeeze(-1)                 # (dA,)
        t = torch.arange(H, device=a_mins.device, dtype=center.dtype)   # (H,)
        decay = beta + (1-beta) * torch.exp(-t / tau)                   # (H,)

        mean = center.unsqueeze(0).expand(H, da)                        # (H, dA)
        standard_dev = decay.unsqueeze(1) * sigma_0.unsqueeze(0)        # (H, dA)

        # Initialize a Time-varying Diagonal Gaussian distribution
        return Normal(
            loc=mean,
            scale=standard_dev.clamp_min(std_min)
        )        
    
    @staticmethod
    def warm_start(
        elites: torch.Tensor, 
        a_mins: torch.Tensor, 
        a_maxs: torch.Tensor, 
        widening_factor=1.5, 
        std_min=1e-6
        ) -> Normal:
        """
        Warm-start the A-space Normal from previous elites.

        Steps
        -----
        1) Compute mean/std across elites.
        2) Shift mu_a, sigma_a forward by one time-step.
        3) Tail:
           - "repeat": use last mean/std
           - "center": set last mean to center of box (keeps last std)
        4) Widen sigma_a[0] by `widening_factor`.
        5) (Optional) Anchor mu_a[0] toward last executed action with convex blend.

        Args:
            elites : (N_e, H, dA)
                Elite **A-space** action sequences from previous iteration.
            a_mins, a_maxs : (dA,1) or (dA,)
                Bounds are not used. These are part of the interface but not required.
            widening_factor : float, default: 1.5
                Multiplier for sigma_a at t=0 after shift.
            std_min : float, default: 1e-6
                Std floor.
        
        Returns:
            Normal
                Warm-started A-space Normal (H, dA).
        """        
        mean = torch.mean(elites, dim=0)
        standard_dev = torch.std(elites, dim=0, unbiased=False)

        # Shift the mean and std to the next time step
        shifted_mean = torch.zeros_like(mean, device=mean.device, dtype=mean.dtype)
        shifted_mean[:-1] = mean[1:]
        shifted_std = torch.zeros_like(standard_dev, device=standard_dev.device, dtype=standard_dev.dtype)
        shifted_std[:-1] = standard_dev[1:]

        # Add tail value as the previous last time step
        shifted_mean[-1] = mean[-1]
        shifted_std[-1] = standard_dev[-1]

        # Widen the standard deviation to encourage exploration
        shifted_std[0] = (shifted_std[0] * widening_factor).clamp_min_(std_min)

        return Normal(
            loc=shifted_mean,
            scale=shifted_std.clamp_min(1e-6)
        )
