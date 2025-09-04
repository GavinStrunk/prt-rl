import torch
from torch.distributions import Normal
from typing import Any, Callable
from prt_rl.model_based.planners.rollout import rollout_action_sequence

def temporal_smooth(x: torch.Tensor, method: str = 'none', rho: float = 0.9, kernel_size: int = 0) -> torch.Tensor:
    """
    Smooth along time for each sample and action dim. X: (N,H,dA).
    method='ou' (EMA), 'conv' (1D kernel), or None.
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
    Cross-Entropy Method (CEM) Planner for Model-Based Reinforcement Learning.

    This planner iteratively samples action sequences, evaluates them, and refines the sampling distribution
    based on the best-performing sequences.

    Args:
        action_mins (torch.Tensor): Minimum values for each action dimension. Shape (action_dim, 1).
        action_maxs (torch.Tensor): Maximum values for each action dimension. Shape (action_dim, 1).
        planning_horizon (int): Number of steps to plan ahead (H).
        num_iterations (int): Number of CEM iterations to perform.
        num_samples (int): Number of action sequences to sample per iteration.
        num_elites (int): Number of top-performing sequences to use for updating the distribution.
        initial_std (float): Initial standard deviation for the Gaussian sampling distribution.
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
        Plan a sequence of actions using the CEM algorithm.

        Args:
            evaluate_fn (callable): A function that takes a tensor of shape (B, H, A) and returns a tensor of shape (B,)
                                    representing the cost or reward of each action sequence.

        Returns:
            torch.Tensor: A tensor of shape (1, A) representing the best action sequence found.
        """
        # Initialize the starting distribution
        self._initialize_distribution()

        for _ in range(self.num_iterations):
            # Sample new action sequences - (N, H, da)
            action_sequences = self.bound_strategy.sample(self.distribution, (self.num_action_sequences,), self.action_mins, self.action_maxs)

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

    def _initialize_distribution(self) -> None:
        """
        Initialize the Time-varying Diagonal Gaussian distribution using the center, broad prior approach.
        
        This sets the initial mean to the center of the action box and the standard deviation to a fraction of the action range. 
        If no action bounds are provided then the mean is set to zero and the standard deviation to one.
        """
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


class TanhSquashBound:
    """
    Sampling/refitting happens in U-space (pre-squash).
    Mapping to actions uses: A = (tanh(U)+1)/2 * (a_max - a_min) + a_min
    """    
    @staticmethod
    def sample(distribution: Normal, shape: torch.Size, a_mins: torch.Tensor, a_maxs: torch.Tensor) -> torch.Tensor:
        # Sample distribution in U-space with shape (N, H, da)
        u_actions = distribution.rsample(shape)

        # Apply temporal smoothing to the u-space actions
        u_smooth = temporal_smooth(u_actions, method='ou', rho=0.9)

        # Convert action from U-space to action space
        a_actions = TanhSquashBound._from_u_space(u_smooth, a_mins, a_maxs)
        return a_actions
    
    @staticmethod
    def refit(elites: torch.Tensor, a_mins: torch.Tensor, a_maxs: torch.Tensor, std_min: float = 1e-6) -> Normal:
        # Convert elite actions to U-space
        u_elites = TanhSquashBound._to_u_space(elites, a_mins, a_maxs)
        mean_u = u_elites.mean(dim=0)
        std_u = u_elites.std(dim=0, unbiased=False).clamp_min(std_min)
        return Normal(loc=mean_u, scale=std_u)
    
    @staticmethod
    def cold_start(H: int, a_mins: torch.Tensor, a_maxs: torch.Tensor, beta: float, tau: float) -> Normal:
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
        sigma_0 = torch.full_like(center, 0.6)                                  # (dA,)
        std_u = decay.unsqueeze(1) * sigma_0.unsqueeze(0)                       # (H, dA)

        return Normal(
            loc=mean_u,
            scale=std_u.clamp_min(1e-6)
        )
    
    @staticmethod
    def warm_start(elites: torch.Tensor, a_mins: torch.Tensor, a_maxs: torch.Tensor, widening_factor=1.5, std_min=1e-6) -> Normal:
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

        # Optional (not implemented): anchor first time step mean to executed action

        return Normal(
            loc=shifted_mean,
            scale=shifted_std.clamp_min(1e-6)
        )
    
    @staticmethod
    def _to_u_space(a_actions, a_mins, a_maxs, epsilon=1e-6):
        y = (2*(a_actions - a_mins) / (a_maxs - a_mins) - 1).clamp(-1 + epsilon, 1 - epsilon)
        # atanh(y) = 0.5*log((1+y)/(1-y))
        return 0.5 * torch.log1p(y) - 0.5 * torch.log1p(-y)
    @staticmethod
    def _from_u_space(u_actions, a_mins, a_maxs):
        y = torch.tanh(u_actions)
        return (y + 1) / 2 * (a_maxs - a_mins) + a_mins
      
class ClipBound:
    @staticmethod
    def sample(distribution: Normal, shape: torch.Size, a_mins: torch.Tensor, a_maxs: torch.Tensor) -> torch.Tensor:
        # Sample distribution in A-space with shape (N, H, da)
        a_actions = distribution.rsample(shape)

        # Apply temporal smoothing to the actions
        a_smooth = temporal_smooth(a_actions, method='ou', rho=0.9)

        # Clip to the action bounds
        actions = torch.clamp(a_smooth, a_mins, a_maxs)
        return actions
    
    @staticmethod
    def refit(elites: torch.Tensor, a_mins: torch.Tensor, a_maxs: torch.Tensor, std_min: float = 1e-6) -> Normal:
        mean = elites.mean(dim=0)
        standard_dev = elites.std(dim=0, unbiased=False).clamp_min(std_min)
        return Normal(loc=mean, scale=standard_dev)
    
    @staticmethod
    def cold_start(H: int, a_mins: torch.Tensor, a_maxs: torch.Tensor, beta: float, tau: float) -> Normal:
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
            scale=standard_dev.clamp_min(1e-6)
        )        
    
    @staticmethod
    def warm_start(elites: torch.Tensor, a_mins: torch.Tensor, a_maxs: torch.Tensor, widening_factor=1.5, std_min=1e-6) -> Normal:
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
