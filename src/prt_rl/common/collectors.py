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
            return policy.predict(state, deterministic=deterministic)

class MetricsTracker:
    """
    Tracks collection metrics and logs ONLY when episodes finish. Counts are in env-steps: one vectorized step across N envs adds N.

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
            done:   Tensor shaped (N,) or (N,1) or scalar/bool-like per env; True indicates episode end.
        """
        r_env = self._sum_per_env(reward)   # (N,)
        d_env = self._to_done_mask(done)    # (N,)

        # Count env-steps (one vector step increments by N)
        n = int(r_env.shape[0])
        self.collected_steps += n

        # Accumulate current episodes per env
        self._cur_reward += r_env.to(self._cur_reward.dtype)
        self._cur_length += 1

        # Global cumulative reward
        self.cumulative_reward += float(r_env.sum().item())

        # Log & reset for any envs that finished this step
        if d_env.any():
            finished = torch.nonzero(d_env, as_tuple=False).view(-1).tolist()
            for i in finished:
                ep_r = float(self._cur_reward[i].item())
                ep_L = int(self._cur_length[i].item())

                self.episode_count += 1
                self.last_episode_reward = ep_r
                self.last_episode_length = ep_L

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
            done (torch.Tensor): The done flags, can be a scalar, 1D tensor, or 2D tensor with last dim of size 1.
        Returns:
            torch.Tensor: A boolean mask indicating which environments are done.
        """
        d = torch.as_tensor(done)
        if d.ndim == 0:
            d = d.view(1)
        if d.ndim > 1 and d.shape[-1] == 1:
            d = d.squeeze(-1)
        return d.bool()

    @staticmethod
    def _sum_per_env(x: torch.Tensor) -> torch.Tensor:
        """
        Sum over trailing dimensions so each environment gets a scalar; returns shape (N,).

        Args:
            x (torch.Tensor): The input tensor to sum over.
        Returns:
            torch.Tensor: A tensor with shape (N,) where N is the number of environments.
        """
        t = torch.as_tensor(x)
        if t.ndim == 0:
            t = t.view(1)
        if t.ndim > 1:
            t = t.sum(dim=tuple(range(1, t.ndim)))
        return t    

class SequentialCollector:
    """
    The Sequential Collector collects experience from a single environment sequentially.

    It resets the environment when the previous experience is done.

    Args:
        env (EnvironmentInterface): The environment to collect experience from.
        logger (Optional[Logger]): Optional logger for logging information. Defaults to a new Logger instance.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 logger: Optional[Logger] = None,
                 ) -> None:
        self.env = env
        self.env_params = env.get_parameters()
        self.logger = logger or Logger.create('blank')
        self.metric = MetricsTracker(num_envs=1, logger=self.logger)
        self.previous_experience = None

    def collect_experience(self,
                           policy: 'BaseAgent | BasePolicy | None' = None,
                           num_steps: int = 1
                           ) -> Dict[str, torch.Tensor]:
        """
        Collects the given number of experiences from the environment using the provided policy. 
        
        Since the experiences are collected sequentially, the output shape is (B, ...) where the batch size is the number of steps collected.

        Args:
            policy (BaseAgent | BasePolicy | None): An agent or policy that takes a state and returns an action.
            num_steps (int): The number of steps to collect experience for. Defaults to 1.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
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

        for _ in range(num_steps):
            # Reset the environment if no previous state
            if self.previous_experience is None or self.previous_experience["done"]:
                state, _ = self.env.reset()
            else:
                state = self.previous_experience["next_state"]

            action, value_est, log_prob = get_action_from_policy(policy, state, self.env_params)

            next_state, reward, done, _ = self.env.step(action)

            # Update the Metrics tracker and logging
            self.metric.update(reward, done)

            states.append(state.squeeze(0))
            actions.append(action.squeeze(0))
            next_states.append(next_state.squeeze(0))
            rewards.append(reward.squeeze(0))
            dones.append(done.squeeze(0))

            self.previous_experience = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }

        return {
            "state": torch.stack(states, dim=0),
            "action": torch.stack(actions, dim=0),
            "next_state": torch.stack(next_states, dim=0),
            "reward": torch.stack(rewards, dim=0),
            "done": torch.stack(dones, dim=0),
        }
    
    def collect_trajectory(self, 
                           policy: 'BaseAgent | BasePolicy | None' = None,
                           num_trajectories: Optional[int] = None,
                           min_num_steps: Optional[int] = None
                           ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Collects a single trajectory from the environment using the provided policy.

        You can specify either a number of trajectories to collect by setting num_trajectories or a minimum number of steps to collect. If the later is specified trajectories will be collect until the step count is reached and the last trajectory will continue collecting until it is done. Return the trajectory with the shape (T, ...), where T is the number of steps in the trajectory.
        Args:
            policy (BaseAgent | BasePolicy | None): An agent or policy that takes a state and returns an action.
            num_trajectories (Optional[int]): The number of trajectories to collect. If None, min_num_steps must be provided. Defaults to None.
            min_num_steps (Optional[int]): The minimum number of steps to collect. If None, num_trajectories must be provided. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected trajectory with keys:
                - 'state': The states collected.
                - 'action': The actions taken.
                - 'next_state': The next states after taking the actions.
                - 'reward': The rewards received.
                - 'done': The done flags indicating if the episode has ended.
        """
        if num_trajectories is None and min_num_steps is None:
            num_trajectories = 1

        if num_trajectories is not None and min_num_steps is not None:
            raise ValueError("Only one of num_trajectories or min_num_steps should be provided, not both.")
        
        trajectories = []
        total_steps = 0
        if num_trajectories is not None:
            for _ in range(num_trajectories):
                trajectory = self._collect_single_trajectory(policy)
                total_steps += trajectory['state'].shape[0]
                trajectories.append(trajectory)
        else:
            # Collect until we reach the minimum number of steps
            total_steps = 0
            while total_steps < min_num_steps:
                trajectory = self._collect_single_trajectory(policy)
                total_steps += trajectory['state'].shape[0]
                trajectories.append(trajectory)

        return trajectories, total_steps

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
        """
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        log_probs = []
        value_estimates = []

        # Reset the environment to start a new trajectory
        state, _ = self.env.reset()

        while True:
            action, value_estimate, log_prob = get_action_from_policy(policy, state, self.env_params)

            next_state, reward, done, _ = self.env.step(action)

            # Update the Metrics tracker and logging
            self.metric.update(reward, done)

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            if value_estimate is not None:
                value_estimates.append(value_estimate)
            if log_prob is not None:
                log_probs.append(log_prob)

            # Update the state
            state = next_state

            if done:
                break

        trajectory = {
            "state": torch.cat(states, dim=0),
            "action": torch.cat(actions, dim=0),
            "next_state": torch.cat(next_states, dim=0),
            "reward": torch.cat(rewards, dim=0),
            "done": torch.cat(dones, dim=0),
        }
        if value_estimates:
            trajectory['value_est'] = torch.cat(value_estimates, dim=0)
        if log_probs:
            trajectory['log_prob'] = torch.cat(log_probs, dim=0)

        return trajectory


class ParallelCollector:
    """
    The Parallel Collector collects experience from multiple environments in parallel.
    It resets the environments when the previous experience is done.

    Args:
        env (EnvironmentInterface): The environment to collect experience from.
        flatten (bool): Whether to flatten the collected experience. If flattened the output shape will be (N*T, ...), but if not flattened it will be (N, T, ...). Defaults to True.
        logger (Optional[Logger]): Optional logger for logging information. Defaults to a new Logger instance.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 flatten: bool = True,
                 logger: Optional[Logger] = None,
                 ) -> None:
        self.env = env
        self.env_params = env.get_parameters()
        self.flatten = flatten
        self.logger = logger or Logger.create('blank')
        self.metric = MetricsTracker(num_envs=self.env.get_num_envs(), logger=self.logger) 
        self.previous_experience = None

    def collect_experience(self,
                           policy: 'BaseAgent | BasePolicy | None' = None,
                           num_steps: int = 1
                           ) -> Dict[str, torch.Tensor]:
        """
        Collects the given number of experiences from the environment using the provided policy.

        Args:
            policy (BaseAgent | BasePolicy | None): An agent or policy that takes a state and returns an action.
            num_steps (int): The number of steps to collect experience for. Defaults to 1.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
                - 'state': The states collected. Shape (T, N, ...), or (N*T, ...) if flattened.
                - 'action': The actions taken. Shape (T, N, ...), or (N*T, ...) if flattened.
                - 'next_state': The next states after taking the actions. Shape (T, N, ...), or (N*T, ...) if flattened.
                - 'reward': The rewards received. Shape (T, N, 1), or (N*T, 1) if flattened.
                - 'done': The done flags indicating if the episode has ended. Shape (T, N, 1), or (N*T, 1) if flattened.
                - 'value_est' (optional): The value estimates from the policy, if applicable.
                - 'log_prob' (optional): The log probabilities of the actions, if applicable.
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

        for _ in range(T):
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

            action, value_estimate, log_prob = get_action_from_policy(policy, state, self.env_params)

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

            # Append the tensors to the lists with shape (N, ...)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done) 

            # If the policy provides value estimates or log probabilities, append them to the lists
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

        experience = {
            "state": states,
            "action": actions,
            "next_state": next_states,
            "reward": rewards,
            "done": dones,
        }
        if value_estimates is not None:
            experience['value_est'] = value_estimates

            # Compute the last value estimate for boostrapping
            _, last_value_estimate, _ = get_action_from_policy(policy, self.previous_experience['next_state'], self.env_params)
            experience['last_value_est'] = last_value_estimate  

        if log_probs is not None:
            experience['log_prob'] = log_probs
        
        return experience
    
    def collect_trajectory(self, 
                        policy: 'BaseAgent | BasePolicy | None' = None,
                        num_trajectories: int = 1,
                        ) -> Tuple[Dict[str, List[torch.Tensor]], int]:
        """
        Collects full trajectories in parallel from the environment using the provided policy.

        Args:
            policy (BaseAgent | BasePolicy | None): The policy or agent to use.
            num_trajectories (int): The total number of complete trajectories to collect.

        Returns:
            Tuple[Dict[str, List[torch.Tensor]], int]: 
                - A dictionary where each key maps to a list of per-trajectory tensors (each tensor has shape (T_i, ...)).
                Keys: 'state','action','next_state','reward','done' and optionally 'value_est','log_prob','length'.
                - The total number of steps collected across all returned trajectories.
        """
        N = self.env.get_num_envs()

        # --- Establish starting states for all envs (respect previous_experience if present) ---
        if self.previous_experience is None:
            state, _ = self.env.reset()
        else:
            # Continue from where we left off, but reset finished envs
            state = self.previous_experience["next_state"]
            prev_done = self.previous_experience["done"]
            # ensure prev_done is 1D bool-like over envs
            if prev_done.ndim > 1 and prev_done.shape[-1] == 1:
                prev_done = prev_done.squeeze(-1)
            for i in range(N):
                if bool(prev_done[i].item() if torch.is_tensor(prev_done[i]) else prev_done[i]):
                    reset_state, _ = self.env.reset_index(i)
                    state[i] = reset_state

        # --- Per-env rolling buffers (lists of per-step tensors) ---
        per_env_buffers = []
        for _ in range(N):
            per_env_buffers.append({
                "state": [],
                "action": [],
                "next_state": [],
                "reward": [],
                "done": [],
                "value_est": [],
                "log_prob": [],
            })

        # --- Outputs: lists of completed trajectories (each a tensor of length T_i) ---
        out: Dict[str, List[torch.Tensor]] = {
            "state": [],
            "action": [],
            "next_state": [],
            "reward": [],
            "done": [],
            # Optional keys will be appended only if present
            "value_est": [],
            "log_prob": [],
            "length": [],  # store as 1D tensor per-trajectory for convenience
        }

        episodes_collected = 0
        total_steps = 0

        # Policy eval/no_grad for deterministic eval (feel free to remove eval() if you manage outside)
        policy_cm = torch.no_grad() if policy is not None else nullcontext()
        with policy_cm:
            while episodes_collected < num_trajectories:
                # --- Act ---
                action, value_estimate, log_prob = get_action_from_policy(policy, state, self.env_params)

                # --- Step ---
                next_state, reward, done, _ = self.env.step(action)

                # Normalize shapes for indexing
                done_view = done
                if done_view.ndim > 1 and done_view.shape[-1] == 1:
                    done_view = done_view.squeeze(-1)

                # --- Append this step to each env's buffer ---
                for i in range(N):
                    per_env_buffers[i]["state"].append(state[i])
                    per_env_buffers[i]["action"].append(action[i])
                    per_env_buffers[i]["next_state"].append(next_state[i])
                    per_env_buffers[i]["reward"].append(reward[i])
                    per_env_buffers[i]["done"].append(done[i])
                    if value_estimate is not None:
                        per_env_buffers[i]["value_est"].append(value_estimate[i])
                    if log_prob is not None:
                        per_env_buffers[i]["log_prob"].append(log_prob[i])
                    total_steps += 1

                # --- Finalize any episodes that ended on this step ---
                for i in range(N):
                    if bool(done_view[i].item() if torch.is_tensor(done_view[i]) else done_view[i]):
                        buf = per_env_buffers[i]

                        # Stack lists into (T_i, ...) tensors
                        traj_state = torch.stack(buf["state"], dim=0)
                        traj_action = torch.stack(buf["action"], dim=0)
                        traj_next_state = torch.stack(buf["next_state"], dim=0)
                        traj_reward = torch.stack(buf["reward"], dim=0)
                        traj_done = torch.stack(buf["done"], dim=0)
                        T_i = traj_reward.shape[0]

                        out["state"].append(traj_state)
                        out["action"].append(traj_action)
                        out["next_state"].append(traj_next_state)
                        out["reward"].append(traj_reward)
                        out["done"].append(traj_done)
                        out["length"].append(torch.as_tensor(T_i))

                        if len(buf["value_est"]) > 0:
                            out["value_est"].append(torch.stack(buf["value_est"], dim=0))
                        if len(buf["log_prob"]) > 0:
                            out["log_prob"].append(torch.stack(buf["log_prob"], dim=0))

                        episodes_collected += 1

                        # Clear buffer for this env (it may start a new episode immediately)
                        per_env_buffers[i] = {
                            "state": [],
                            "action": [],
                            "next_state": [],
                            "reward": [],
                            "done": [],
                            "value_est": [],
                            "log_prob": [],
                        }

                        # If we still need more episodes, reset this env so it can keep collecting
                        if episodes_collected < num_trajectories:
                            reset_state, _ = self.env.reset_index(i)
                            next_state[i] = reset_state  # ensure we start next ep from reset state

                # Prepare for next loop and update previous_experience
                self.previous_experience = {
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                    "done": done,
                }
                state = next_state

        # Remove optional keys if none were collected (cleaner downstream)
        if len(out["value_est"]) == 0:
            out.pop("value_est")
        if len(out["log_prob"]) == 0:
            out.pop("log_prob")

        return out, total_steps
