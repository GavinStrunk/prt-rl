"""
Collectors gather experience from environments using the provided policy/agent.
"""
import contextlib
import torch
from typing import Dict, Optional, List, Tuple, Any
from prt_rl.env.interface import EnvironmentInterface, EnvParams, MultiAgentEnvParams
from prt_rl.common.loggers import Logger
from prt_rl.common.policies import BasePolicy

def random_action(env_params: EnvParams, state: torch.Tensor) -> torch.Tensor:
    """
    Randomly samples an action from action space.

    Args:
        env_params (EnvParams): The environment parameters containing action space information.
        state (torch.Tensor): The current state of the environment.

    Returns:
        torch.Tensor: A tensor containing the sampled action.
    """
    device = state.device
    dtype = state.dtype

    if isinstance(env_params, EnvParams):
        ashape = (state.shape[0], env_params.action_len)
        params = env_params
    elif isinstance(env_params, MultiAgentEnvParams):
        ashape = (state.shape[0], env_params.num_agents, env_params.agent.action_len)
        params = env_params.agent
    else:
        raise ValueError("env_params must be a EnvParams or MultiAgentEnvParams")
    
    if not params.action_continuous:
        # Add 1 to the high value because randint samples between low and 1 less than the high: [low,high)
        action = torch.randint(low=params.action_min, high=params.action_max + 1,
                               size=ashape, dtype=torch.long, device=device)
    else:
        action = torch.rand(size=ashape, dtype=dtype, device=device)
        # Scale the random [0,1] actions to the action space [min,max]
        max_actions = torch.tensor(params.action_max).unsqueeze(0)
        min_actions = torch.tensor(params.action_min).unsqueeze(0)
        action = action * (max_actions - min_actions) + min_actions

    return action 

def get_action_from_policy(
        policy, 
        state: torch.Tensor, 
        env_params: EnvParams = None,
        deterministic: bool = False,
        inference_mode: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get an action from the policy given the state.

    Unified policy interface:
      - If `policy` implements .predict(obs, deterministic=False) -> (action, value, log_prob),
        we call that.
      - If `policy` is None: fall back to random_action(...).
      - Else: treat it as a callable that returns only the action.
    
    Args:
        policy: The policy to get the action from.
        state (torch.Tensor): The current state of the environment.
        env_params (EnvParams, optional): The environment parameters. Required if policy is None so a random action can be taken.
    
    Returns:
        Tuple: 
            - action (torch.Tensor): The action to take. Shape (B, action_dim)
            - value_estimate (torch.Tensor): The value estimate from the policy. Shape (B, 1) if applicable otherwise None.
            - log_prob (torch.Tensor): The log probability of the action. Shape (B, 1) if applicable otherwise None.
    """
    ctx = torch.no_grad() if inference_mode else contextlib.nullcontext()
    with ctx:
        if policy is None:
            return random_action(env_params, state), None, None
        else:
            prediction = policy.predict(state, deterministic=deterministic)

            # If only the action is returned then set the value estimate and log probs to None
            if len(prediction) == 1:
                prediction = prediction, None, None
            return prediction

class MetricsTracker:
    """
    Tracks collection metrics and logs ONLY when episodes finish. Counts are in env-steps: one vectorized step across N envs adds N.

    .. note::
        This class is designed to be used with single or vectorized environments. If multiple environments emit done on the same step, an episode reward will be logged for each environment with the same environment step value.

    Args:
        num_envs (int): The number of environments being tracked.
        logger (Logger | None): Optional logger for logging metrics. If None, no logging is performed.
    """
    def __init__(self, 
                 num_envs: int, 
                 logger: "Logger | None" = None
                 ) -> None:
        self.num_envs = int(num_envs)
        self.logger = logger

        # Global counters
        self.collected_steps: int = 0           # env-steps
        self.cumulative_reward: float = 0.0
        self.episode_count: int = 0
        self.last_episode_reward: float = 0.0
        self.last_episode_length: int = 0

        # Per-env episode accumulators
        self._cur_reward = torch.zeros(self.num_envs, dtype=torch.float32)
        self._cur_length = torch.zeros(self.num_envs, dtype=torch.int64)

    def reset(self) -> None:
        """
        Reset all counters and accumulators.
        """
        self.collected_steps = 0
        self.cumulative_reward = 0.0
        self.episode_count = 0
        self.last_episode_reward = 0.0
        self.last_episode_length = 0
        self._cur_reward.zero_()
        self._cur_length.zero_()

    def update(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        """
        Update metrics for a single environment step (vectorized over N).

        Args:
            reward: Tensor shaped (N, 1) or (…,) whose trailing dims will be summed per env.
            done:   Tensor shaped (N, 1) or scalar/bool-like per env; True indicates episode end.
        """
        # Move reward and done to CPU if they are not already there
        reward = reward.cpu()
        done = done.cpu()
        
        # Ensure reward and done are tensors with shape (N,)
        r_env = self._sum_rewards_per_env(reward) 
        d_env = self._to_done_mask(done)

        # Count env-steps (one vector step increments by N)
        n = int(r_env.shape[0])
        self.collected_steps += n

        # Accumulate current episodes rewards and lengths per env
        self._cur_reward += r_env.to(self._cur_reward.dtype)
        self._cur_length += 1

        # Global cumulative reward
        self.cumulative_reward += float(r_env.sum().item())

        # Log & reset for any envs that finished this step
        if d_env.any():
            # Get a list of environment indexes that are done
            finished = torch.nonzero(d_env, as_tuple=False).view(-1).tolist()

            for i in finished:
                # Compute the epsode reward and length for this env
                ep_r = float(self._cur_reward[i].item())
                ep_L = int(self._cur_length[i].item())

                # Increment the global episode count and save as most recent or last episode metrics
                self.episode_count += 1
                self.last_episode_reward = ep_r
                self.last_episode_length = ep_L

                # Log the episode metrics if a logger is provided
                if self.logger is not None:
                    step = self.collected_steps
                    self.logger.log_scalar("episode_reward", ep_r, iteration=step)
                    self.logger.log_scalar("episode_length", ep_L, iteration=step)
                    self.logger.log_scalar("cumulative_reward", float(self.cumulative_reward), iteration=step)
                    self.logger.log_scalar("episode_number", float(self.episode_count), iteration=step)

                # Clear accumulators for that env
                self._cur_reward[i] = 0.0
                self._cur_length[i] = 0
    
    @staticmethod
    def _to_done_mask(done: torch.Tensor) -> torch.Tensor:
        """
        Convert done flags to a boolean mask.

        Args:
            done (torch.Tensor): The done flags, can be a scalar, 1D tensor (N,), or 2D tensor with last dim of size 1 (N, 1).
        Returns:
            torch.Tensor: A boolean mask indicating which environments are done with shape (N,).
        """
        d = torch.as_tensor(done)
        if d.ndim == 0:
            d = d.view(1)
        if d.ndim > 1 and d.shape[-1] == 1:
            d = d.squeeze(-1)
        return d.bool()

    @staticmethod
    def _sum_rewards_per_env(reward: torch.Tensor) -> torch.Tensor:
        """
        Sum over trailing dimensions so each environment gets a scalar; returns shape (N,).

        Args:
            reward (torch.Tensor): The input tensor to sum over. Rewards can be scalar, 1D tensor (N,), 2D tensor with last dim of size 1 (N, 1), or 3D tensor (N, D, 1) .
        Returns:
            torch.Tensor: A tensor with shape (N,) where N is the number of environments.
        """
        t = torch.as_tensor(reward)
        if t.ndim == 0:
            t = t.view(1)
        if t.ndim > 1:
            t = t.sum(dim=tuple(range(1, t.ndim)))
        return t    

class SequentialCollector:
    """
    The Sequential Collector collects experience from a single environment sequentially.

    The sequential collector can collect experiences which returns a specific number of environment steps or specific number of trajectories. If you are collecting experience and the environment is done, but the number of steps is not reached, the environment is reset and continues collecting. 
    Once the number of steps is reached, collection stops so a partial trajectory is likely for the last trajectory when collecting experiences. When you are collecting trajectories, you can either specific a number of trajectories and the collector will keep collecting until that number of trajectories is reached or you can specific a minimum number of steps and it will collect trajectories until the steps are reached and then continue until the trajectory finishes. You will not get partial trajectories in this mode, but the number of steps will vary on subsequent calls.

    .. note::
        Do not collect trajectories with an environment that never ends (i.e. done is never True) as the collector will never return. In this case collect experiences instead.

    Args:
        env (EnvironmentInterface): The environment to collect experience from.
        logger (Logger | None): Optional logger for logging information. Defaults to a blank Logger instance.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 logger: Logger | None = None,
                 ) -> None:
        self.env = env
        self.env_params = env.get_parameters()
        self.logger = logger if logger is not None else Logger.create('blank')
        self.metric_tracker = MetricsTracker(num_envs=1, logger=self.logger)
        self.previous_experience = None

    def collect_experience(self,
                           policy: 'BaseAgent | BasePolicy | None' = None,
                           num_steps: int = 1,
                           bootstrap: bool = True
                           ) -> Dict[str, torch.Tensor]:
        """
        Collects the given number of environment steps using the provided policy. 
        
        Since the experiences are collected sequentially, the output shape is (B, ...) where the batch size, B, is equal to the number of time steps, T, collected. This method collects exactly the number of steps specified, so it is possible to get multiple trajectories and the last one can be a partial trajectory.

        Args:
            policy (BaseAgent | BasePolicy | None): An agent or policy that takes a state and returns an action.
            num_steps (int): The number of steps to collect experience for. Defaults to 1.
            bootstrap (bool): Whether to compute the last value estimate V(s_{T+1}) for bootstrapping if the last step is not done and the policy provides value estimates. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys (B=T):
                - 'state': The states collected. Shape (B, state_dim)
                - 'action': The actions taken. Shape (B, action_dim)
                - 'next_state': The next states after taking the actions. Shape (B, state_dim)
                - 'reward': The rewards received. Shape (B, 1)
                - 'done': The done flags indicating if the episode has ended. Shape (B, 1)
                - 'value_est' (optional): The value estimates from the policy, if applicable. Shape (B, 1)
                - 'log_prob' (optional): The log probabilities of the actions, if applicable. Shape (B, 1)
                - 'last_value_est' (optional): The last value estimate V(s_{T+1}) for bootstrapping, if applicable. Shape (1, 1)
        """
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        value_estimates = []
        log_probs = []
        last_value_estimate = None

        for _ in range(num_steps):
            # Collect a single step
            state, action, next_state, reward, done, value_est, log_prob = self._collect_step(policy)

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            if value_est is not None:
                value_estimates.append(value_est)
            if log_prob is not None:
                log_probs.append(log_prob)

        # If the last step was not done and value estimates are available, then compute the last value estimate for bootstrapping
        if not self.previous_experience['done'] and value_estimates and bootstrap:
            _, last_value_estimate, _ = get_action_from_policy(policy, self.previous_experience['next_state'], self.env_params)

        exp = {
            "state": torch.stack(states, dim=0),
            "action": torch.stack(actions, dim=0),
            "next_state": torch.stack(next_states, dim=0),
            "reward": torch.stack(rewards, dim=0),
            "done": torch.stack(dones, dim=0),
        }

        # Add the optional keys only if they were collected
        if value_estimates:
            exp['value_est'] = torch.stack(value_estimates, dim=0)
        if log_probs:
            exp['log_prob'] = torch.stack(log_probs, dim=0)
        if last_value_estimate is not None:
            exp['last_value_est'] = last_value_estimate

        return exp
    
    def collect_trajectory(self, 
                        policy: 'BaseAgent | BasePolicy | None' = None,
                        num_trajectories: int | None = None,
                        min_num_steps: int | None = None
                        ) -> Dict[str, torch.Tensor]:
        """
        Collects one or more full trajectories and returns a single stacked/batched dictionary.

        You can specify either a number of trajectories to collect via `num_trajectories`
        or a minimum number of steps via `min_num_steps`. If `min_num_steps` is provided,
        trajectories are collected until the step count is reached, and the last trajectory
        is completed (no partials). Returns tensors stacked along dim=0 with shape (B, …),
        where B is the total number of steps across all collected trajectories.

        Args:
            policy (BaseAgent | BasePolicy | None): Policy used to act in the environment.
            num_trajectories (int | None): Number of full trajectories to collect.
            min_num_steps (int | None): Minimum total steps to collect (last trajectory finished).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected trajectory with keys:
                - 'state': The states collected. Shape (B, state_dim)
                - 'action': The actions taken. Shape (B, action_dim)
                - 'next_state': The next states after taking the actions. Shape (B, state_dim)
                - 'reward': The rewards received. Shape (B, 1)
                - 'done': The done flags indicating if the episode has ended. Shape (B, 1)
                - 'value_est' (optional): The value estimates from the policy, if applicable. Shape (B, 1)
                - 'log_prob' (optional): The log probabilities of the actions, if applicable. Shape (B, 1)
        """
        if num_trajectories is None and min_num_steps is None:
            num_trajectories = 1
        if num_trajectories is not None and min_num_steps is not None:
            raise ValueError("Only one of num_trajectories or min_num_steps should be provided, not both.")

        # Accumulators (store tensors per trajectory; one cat per key at the end)
        states, actions, next_states, rewards, dones = [], [], [], [], []
        value_ests, log_probs = [], []
        total_steps = 0

        if num_trajectories is not None:
            for _ in range(num_trajectories):
                traj = self._collect_single_trajectory(policy)

                states.append(traj['state'])
                actions.append(traj['action'])
                next_states.append(traj['next_state'])
                rewards.append(traj['reward'])
                dones.append(traj['done'])
                ve = traj.get('value_est', None)
                lp = traj.get('log_prob', None)

                if ve is not None: 
                    value_ests.append(ve)
                if lp is not None: 
                    log_probs.append(lp)

                total_steps += traj['state'].shape[0]
        else:
            # min_num_steps mode: keep collecting full trajectories until threshold reached
            target = int(min_num_steps)
            while total_steps < target:
                traj = self._collect_single_trajectory(policy)

                states.append(traj['state'])
                actions.append(traj['action'])
                next_states.append(traj['next_state'])
                rewards.append(traj['reward'])
                dones.append(traj['done'])
                ve = traj.get('value_est', None)
                lp = traj.get('log_prob', None)

                if ve is not None: 
                    value_ests.append(ve)
                if lp is not None: 
                    log_probs.append(lp)

                total_steps += traj['state'].shape[0]

        # Concatenate once per key -> (B, ...)
        out = {
            "state":      torch.cat(states, dim=0) if len(states) else torch.empty(0),
            "action":     torch.cat(actions, dim=0) if len(actions) else torch.empty(0),
            "next_state": torch.cat(next_states, dim=0) if len(next_states) else torch.empty(0),
            "reward":     torch.cat(rewards, dim=0) if len(rewards) else torch.empty(0),
            "done":       torch.cat(dones, dim=0) if len(dones) else torch.empty(0),
        }

        # Add the optional keys only if they were collected
        if value_ests:
            out["value_est"] = torch.cat(value_ests, dim=0)
        if log_probs:
            out["log_prob"]  = torch.cat(log_probs,  dim=0)

        return out
    
    def get_metric_tracker(self) -> MetricsTracker:
        """
        Returns the internal MetricsTracker instance for accessing collection metrics.

        Returns:
            MetricsTracker: The internal MetricsTracker instance.
        """
        return self.metric_tracker

    def _collect_single_trajectory(self, 
                                   policy: 'BaseAgent | BasePolicy | None' = None
                                   ) -> Dict[str, torch.Tensor]:
        """
        Collects a single trajectory from the environment using the provided policy.

        Args:
            policy (BaseAgent | BasePolicy | None): An agent or policy that takes a state and returns an action.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected trajectory with keys:
                - 'state': The states collected. Shape (T, state_dim)
                - 'action': The actions taken. Shape (T, action_dim)
                - 'next_state': The next states after taking the actions. Shape (T, state_dim)
                - 'reward': The rewards received. Shape (T, 1)
                - 'done': The done flags indicating if the episode has ended. Shape (T, 1)
                - 'value_est' (optional): The value estimates from the policy, if applicable. Shape (T, 1)
                - 'log_prob' (optional): The log probabilities of the actions, if applicable. Shape (T, 1)
        """
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        value_estimates = []
        log_probs = []

        while True:
            # Collect a single step
            state, action, next_state, reward, done, value_est, log_prob = self._collect_step(policy)

            # Append the step to the trajectory
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            if value_est is not None:
                value_estimates.append(value_est)
            if log_prob is not None:
                log_probs.append(log_prob)

            # If the episode is done, break the loop because we have a full trajectory
            if done:
                break

        traj = {
            "state": torch.stack(states, dim=0),
            "action": torch.stack(actions, dim=0),
            "next_state": torch.stack(next_states, dim=0),
            "reward": torch.stack(rewards, dim=0),
            "done": torch.stack(dones, dim=0),
        }

        # Add the optional keys only if they were collected
        if value_estimates:
            traj['value_est'] = torch.stack(value_estimates, dim=0)
        if log_probs:
            traj['log_prob'] = torch.stack(log_probs, dim=0)

        return traj
    
    def _collect_step(self,
                      policy: 'BaseAgent | BasePolicy | None' = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collects a single step from the environment using the provided policy.
        
        Args:
            policy (BaseAgent | BasePolicy | None): An agent or policy that takes a state and returns an action.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - state: The current state of the environment. Shape (state_dim,)
                - action: The action taken by the policy. Shape (action_dim,)
                - next_state: The next state after taking the action. Shape (state_dim,)
                - reward: The reward received from the environment. Shape (1,)
                - done: The done flag indicating if the episode has ended. Shape (1,)
                - value_estimate: The value estimate from the policy, if applicable. Shape (1,) or None.
                - log_prob: The log probability of the action, if applicable. Shape (1,) or None.
        """
        # Reset the environment if no previous state
        if self.previous_experience is None or self.previous_experience["done"]:
            state, _ = self.env.reset()
        else:
            state = self.previous_experience["next_state"]

        action, value_est, log_prob = get_action_from_policy(policy, state, self.env_params)
        next_state, reward, done, _ = self.env.step(action)

        # Update the Metrics tracker and logging
        self.metric_tracker.update(reward, done)

        # Save the step for the next step
        self.previous_experience = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }

        # Convert tensors from shape (1, ...) to (...)
        state = state.squeeze(0)
        action = action.squeeze(0)
        next_state = next_state.squeeze(0)
        reward = reward.squeeze(0)
        done = done.squeeze(0)
        if value_est is not None:
            value_est = value_est.squeeze(0)
        if log_prob is not None:
            log_prob = log_prob.squeeze(0)

        return state, action, next_state, reward, done, value_est, log_prob

class ParallelCollector:
    """
    The Parallel Collector collects experience from multiple environments in parallel.

    The parallel collector can collect experiences which returns a specific number of environment steps or specific number of trajectories. If you are collecting experience and the environment is done, but the number of steps is not reached, the environment is reset and continues collecting.

    .. note::
        Do not collect trajectories with an environment that never ends (i.e. done is never True) as the collector will never return. In this case collect experiences instead.

    Args:
        env (EnvironmentInterface): The environment to collect experience from.
        logger (Logger | None): Optional logger for logging information. Defaults to a new Logger instance.
        flatten (bool): Whether to flatten the collected experience. If flattened the output shape will be (N*T, ...), but if not flattened it will be (N, T, ...). Defaults to True.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 logger: Logger | None = None,
                 flatten: bool = True,
                 ) -> None:
        self.env = env
        self.env_params = env.get_parameters()
        self.flatten = flatten
        self.logger = logger if logger is not None else Logger.create('blank')
        self.metric = MetricsTracker(num_envs=self.env.get_num_envs(), logger=self.logger) 
        self.previous_experience = None

    def collect_experience(self,
                           policy: 'BaseAgent | BasePolicy | None' = None,
                           num_steps: int = 1,
                           bootstrap: bool = True
                           ) -> Dict[str, torch.Tensor]:
        """
        Collects the given number of experiences from the environment using the provided policy.

        The experiences are collected across all environments, so the actual number of steps is ceil(num_steps / N) where N is the number of environments. The output shape is (T, N, ...) if not flattened, or (N*T, ...) if flattened. 
        
        Args:
            policy (BaseAgent | BasePolicy | None): An agent or policy that takes a state and returns an action.
            num_steps (int): The number of steps to collect experience for. Defaults to 1.
            bootstrap (bool): Whether to compute the last value estimate V(s_{T+1}) for bootstrapping if the last step is not done and the policy provides value estimates. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
                - 'state': The states collected. Shape (T, N, ...), or (N*T, ...) if flattened.
                - 'action': The actions taken. Shape (T, N, ...), or (N*T, ...) if flattened.
                - 'next_state': The next states after taking the actions. Shape (T, N, ...), or (N*T, ...) if flattened.
                - 'reward': The rewards received. Shape (T, N, 1), or (N*T, 1) if flattened.
                - 'done': The done flags indicating if the episode has ended. Shape (T, N, 1), or (N*T, 1) if flattened.
                - 'value_est' (optional): The value estimates from the policy, if applicable. Shape (T, N, 1), or (N*T, 1) if flattened.
                - 'log_prob' (optional): The log probabilities of the actions, if applicable. Shape (T, N, 1), or (N*T, 1) if flattened.
                - 'last_value_est' (optional): The last value estimate for bootstrapping, if applicable. (N, 1)
        """
        # Get the number of steps to take per environment to get at least `num_steps`
        # A trick for ceiling division: (a + b - 1) // b
        N = self.env.get_num_envs()
        T = (num_steps + N - 1) // N

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        value_estimates = []
        log_probs = []
        last_value_estimate = None

        for _ in range(T):
            # Collect a single step
            state, action, next_state, reward, done, value_estimate, log_prob = self._collect_step(policy)

            # Append the tensors to the lists with shape (N, ...)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done) 
            if value_estimate is not None:
                value_estimates.append(value_estimate)
            if log_prob is not None:
                log_probs.append(log_prob)

        if self.flatten:
            # Concatenate the lists of tensors into a single tensor with shape (N*T, ...)
            states = torch.cat(states, dim=0)
            actions = torch.cat(actions, dim=0)
            next_states = torch.cat(next_states, dim=0)
            rewards = torch.cat(rewards, dim=0)
            dones = torch.cat(dones, dim=0)
            value_estimates = torch.cat(value_estimates, dim=0) if value_estimates else None
            log_probs = torch.cat(log_probs, dim=0) if log_probs else None
        else:
            # Stack the lists of tensors into a single tensor with shape (T, N, ...)
            states = torch.stack(states, dim=0)
            actions = torch.stack(actions, dim=0)
            next_states = torch.stack(next_states, dim=0)
            rewards = torch.stack(rewards, dim=0)
            dones = torch.stack(dones, dim=0)
            value_estimates = torch.stack(value_estimates, dim=0) if value_estimates else None
            log_probs = torch.stack(log_probs, dim=0) if log_probs else None

        # If the last step was not done in any environment and value estimates are available, then compute the last value estimate for bootstrapping
        if value_estimates is not None and bootstrap:
            _, last_value_estimate, _ = get_action_from_policy(policy, self.previous_experience['next_state'], self.env_params)
        
        exp = {
            "state": states,
            "action": actions,
            "next_state": next_states,
            "reward": rewards,
            "done": dones,
        }

        # Add the optional keys only if they were collected
        if value_estimates is not None:
            exp['value_est'] = value_estimates
        if log_probs is not None:
            exp['log_prob'] = log_probs
        if last_value_estimate is not None:
            exp['last_value_est'] = last_value_estimate

        return exp
    
    def collect_trajectory(self, 
                        policy: 'BaseAgent | BasePolicy | None' = None,
                        num_trajectories: int | None = None,
                        min_num_steps: int | None = None
                        ) -> Tuple[Dict[str, List[torch.Tensor]], int]:
        """
        Collects full trajectories in parallel from the environment using the provided policy.

        If the number of trajectories specified matches the number of environments, it will collect one trajectory from each environment. 
        If the number of trajectories is less than the number of environments, it will collect the specified number of trajectories from the first N environments. 
        If the number of trajectories is greater than the number of environments, it will collect num_trajectories // N trajectories from each environment, where N is the number of environments, 
        and then get the remaining trajectories from whichiever environments complete first. 

        The output is a dictionary with keys (state, action, next_state, reward, done) where each key contains a tensor with the first dimension (B, ...) where B is the sum of each trajectories timesteps T.

        Args:
            policy (BaseAgent | BasePolicy | None): The policy or agent to use.
            num_trajectories (int | None): The total number of complete trajectories to collect.
            min_num_steps (int | None): The minimum number of steps to collect before completing the trajectories. If specified, will collect until the minimum number of steps is reached, then complete the last trajectory.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
                - state: The current state of the environment. Shape (B, state_dim)
                - action: The action taken by the policy. Shape (B, action_dim)
                - next_state: The next state after taking the action. Shape (B, state_dim)
                - reward: The reward received from the environment. Shape (B, 1)
                - done: The done flag indicating if the episode has ended. Shape (B, 1)
                - value_estimate: The value estimate from the policy, if applicable. Shape (B, 1) or None.
                - log_prob: The log probability of the action, if applicable. Shape (B, 1) or None.
        """
        # Helper to check if we have enough trajectories collected

                
        if num_trajectories is None and min_num_steps is None:
            num_trajectories = 1
        if num_trajectories is not None and min_num_steps is not None:
            raise ValueError("Only one of num_trajectories or min_num_steps should be provided, not both.")

        # Get the number of environments
        N = int(self.env.get_num_envs())

        # Force fresh full episodes
        self.previous_experience = None

        # Per-step accumulators (N-first); we'll stack to (T, N, ...) once.
        states_steps, actions_steps, next_states_steps = [], [], []
        rewards_steps, dones_steps = [], []
        value_steps, logp_steps = [], []

        # Bookkeeping for stopping criteria while stepping
        per_env_counts = [0] * N                # #completed episodes per env
        cur_ep_len = torch.zeros(N, dtype=torch.long)
        completed_steps_sum = 0                 # only counts lengths of completed episodes
        completed_total = 0

        while True:
            state, action, next_state, reward, done, value_est, log_prob = self._collect_step(policy)

            states_steps.append(state)           # (N, *state_shape)
            actions_steps.append(action)         # (N, action_len)
            next_states_steps.append(next_state) # (N, *state_shape)
            rewards_steps.append(reward)         # (N, 1)
            dones_steps.append(done)             # (N, 1) bool
            if value_est is not None:
                value_steps.append(value_est)    # (N, 1)
            if log_prob is not None:
                logp_steps.append(log_prob)      # (N, 1)

            # Update episode counters
            done_bool = done.view(-1).to(torch.bool)  # (N,)
            finished_idx = done_bool.nonzero(as_tuple=False).view(-1)
            n_finished = int(finished_idx.numel())
            completed_total += n_finished

            # update per-env counts and completed_steps_sum (for min_num_steps)
            cur_ep_len += 1
            if n_finished > 0:
                for i in finished_idx.tolist():
                    per_env_counts[i] += 1
                    completed_steps_sum += int(cur_ep_len[i].item())
                    cur_ep_len[i] = 0

            # Stop conditions
            if num_trajectories is not None:
                if self._have_enough_trajectories(int(num_trajectories), N, per_env_counts):
                    break
            else:
                if completed_steps_sum >= int(min_num_steps):
                    break

        # Nothing recorded (edge case)
        if len(states_steps) == 0:
            raise ValueError("No steps were collected. Ensure the environment is properly configured and the policy is valid.")

        # Stack to (T, N, ...)
        states_TN      = torch.stack(states_steps,      dim=0)  # (T,N,*state_shape)
        actions_TN     = torch.stack(actions_steps,     dim=0)  # (T,N,action_len)
        next_states_TN = torch.stack(next_states_steps, dim=0)  # (T,N,*state_shape)
        rewards_TN     = torch.stack(rewards_steps,     dim=0)  # (T,N,1)
        dones_TN       = torch.stack(dones_steps,       dim=0)  # (T,N,1) bool
        values_TN = torch.stack(value_steps, dim=0) if value_steps else None
        logp_TN = torch.stack(logp_steps,  dim=0) if logp_steps else None

        # ---- Build per-env episode segments with completion times ----
        done_mask = dones_TN.squeeze(-1)  # (T,N) bool

        # Per-env ordered lists of segments (by completion time within the env)
        per_env_segments = [[] for _ in range(N)]
        # Also a global list for min_num_steps path
        global_segments = []

        # Loop through the environments in done mask to find episode segments
        for env_i in range(N):
            # Check if this env finished any episodes and get the timestep indexes if so
            done_idx = torch.nonzero(done_mask[:, env_i], as_tuple=False).flatten()
            if done_idx.numel() == 0:
                continue

            # Build segments: (env_i, t_start, t_end)
            prev_end = -1
            for t_end in done_idx.tolist():
                start = prev_end + 1
                # length = t_end - start + 1
                seg = (env_i, start, t_end)  # include ordinal within-env
                per_env_segments[env_i].append(seg)
                global_segments.append(seg)
                prev_end = t_end

        # Nothing finished -> return empty
        if not global_segments:
            raise ValueError("No complete episodes were recorded. Ensure the environment is properly configured and the policy is valid.")

        selected = []
        if num_trajectories is not None:
            K = int(num_trajectories)
            base = K // N
            rem  = K % N

            # (1) Take earliest `base` episodes per env (by within-env order)
            for env_i in range(N):
                segs = per_env_segments[env_i]
                if base > 0 and len(segs) >= base:
                    selected.extend(segs[:base])
                elif base > 0:
                    # Shouldn't happen due to stopping rule, but guard anyway
                    selected.extend(segs)

            # (2) Candidates for remainder: the (base)-th episode (0-based) of each env,
            #     i.e., the first "extra" beyond the base. Sort by global finish time;
            #     pick earliest `rem` from DISTINCT envs (by construction they are distinct).
            if rem > 0:
                candidates = []
                for env_i in range(N):
                    segs = per_env_segments[env_i]
                    if len(segs) >= base + 1:
                        # This env's extra episode candidate
                        candidates.append(segs[base])  # (env_i, start, end)
                
                # Sort candidates by end time and take earliest `rem`
                candidates.sort(key=lambda s: s[-1])
                selected.extend(candidates[:rem])

            # Order selected by global finish time to preserve temporal ordering
            selected.sort(key=lambda s: s[-1])

        else:
            # min_num_steps: just take earliest episodes by global completion time until sum lengths >= target
            need = int(min_num_steps)
            acc = 0
            global_segments.sort(key=lambda s: s[-1])
            for seg in global_segments:
                selected.append(seg)

                # Add the length of this segment to the accumulator
                acc += seg[-1] - seg[-2]
                if acc >= need:
                    break

        # ---- Slice and concatenate selected segments into (B, ...) ----
        cat_states, cat_actions, cat_next_states = [], [], []
        cat_rewards, cat_dones = [], []
        cat_values, cat_logp = [], []

        for (env_i, t0, t1) in selected:
            cat_states.append(     states_TN[t0:t1+1, env_i, ...])
            cat_actions.append(    actions_TN[t0:t1+1, env_i, ...])
            cat_next_states.append(next_states_TN[t0:t1+1, env_i, ...])
            cat_rewards.append(    rewards_TN[t0:t1+1, env_i, ...])
            cat_dones.append(      dones_TN[t0:t1+1, env_i, ...])
            if values_TN is not None:
                cat_values.append(values_TN[t0:t1+1, env_i, ...])
            if logp_TN is not None:
                cat_logp.append(logp_TN[t0:t1+1, env_i, ...])

        if len(cat_states) == 0:
            raise ValueError("No complete episodes were recorded. Ensure the environment is properly configured and the policy is valid.")

        out = {
            "state":      torch.cat(cat_states,      dim=0),
            "action":     torch.cat(cat_actions,     dim=0),
            "next_state": torch.cat(cat_next_states, dim=0),
            "reward":     torch.cat(cat_rewards,     dim=0),
            "done":       torch.cat(cat_dones,       dim=0),
        }

        if len(cat_values) > 0:
            out["value_est"] = torch.cat(cat_values, dim=0) 
        if len(cat_logp) > 0:
            out["log_prob"]  = torch.cat(cat_logp,   dim=0)

        return out 

    def get_metric_tracker(self) -> MetricsTracker:
        """
        Returns the internal MetricsTracker instance for accessing collection metrics.

        Returns:
            MetricsTracker: The internal MetricsTracker instance.
        """
        return self.metric  
    
    @staticmethod
    def _have_enough_trajectories(K: int, N: int, per_env_counts: list) -> bool:
        """
        Check if we have collected at least K full trajectories across all environments.
        
        Args:
            K (int): The target number of trajectories to collect.
        Returns:
            bool: True if we have collected at least K trajectories, False otherwise.
        """
        divisible_episodes = K // N
        remainder_episodes  = K % N

        # Continue collecting until there are a divisible number of episodes per env
        if any(c < divisible_episodes for c in per_env_counts):
            return False
        
        # If the number of desired episodes is a multiple of N, there are not remainder episodes
        if remainder_episodes == 0:
            return True
        
        # Need at least `rem` envs with one extra (>= base+1)
        extras = sum(1 for c in per_env_counts if c >= divisible_episodes + 1)
        return extras >= remainder_episodes    

    
    def _collect_step(self,
                      policy: 'BaseAgent | BasePolicy | None' = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: 
        """
        Collects a single step from the environment using the provided policy.
        
        Args:
            policy (BaseAgent | BasePolicy | None): An agent or policy that takes a state and returns an action.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                - state: The current state of the environment. Shape (N, state_dim)
                - action: The action taken by the policy. Shape (N, action_dim)
                - next_state: The next state after taking the action. Shape (N, state_dim)
                - reward: The reward received from the environment. Shape (N, 1)
                - done: The done flag indicating if the episode has ended. Shape (N, 1)
                - value_estimate: The value estimate from the policy, if applicable. Shape (N, 1) or None.
                - log_prob: The log probability of the action, if applicable. Shape (N, 1) or None.
        """
        # Reset the environment if no previous state
        if self.previous_experience is None:
            state, _ = self.env.reset()
        else:
            # Only reset the environments that are done
            state = self.previous_experience["next_state"]
            for i in range(self.previous_experience["done"].shape[0]):
                if self.previous_experience["done"][i]:
                    # Reset the environment for this index
                    reset_state, _ = self.env.reset_index(i)
                    # Update the previous experience for this index
                    state[i] = reset_state

        action, value_est, log_prob = get_action_from_policy(policy, state, self.env_params)

        # Step the environment with the action
        next_state, reward, done, _ = self.env.step(action)

        # Update the Metrics tracker and logging
        self.metric.update(reward, done)

        # Save the previous experience for the next step
        self.previous_experience = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }

        return state, action, next_state, reward, done, value_est, log_prob                
    
