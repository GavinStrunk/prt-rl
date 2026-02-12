"""
Collectors gather experience from environments using the provided policy/agent.
"""
import torch
from typing import Dict, Optional, List, Tuple, Any
from prt_rl.env.interface import EnvironmentInterface, EnvParams, MultiAgentEnvParams
from prt_rl.common.loggers import Logger
from prt_rl.common.policies import Policy


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


class Collector:
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
        self.logger = logger if logger is not None else Logger()
        self.metric = MetricsTracker(num_envs=self.env.get_num_envs(), logger=self.logger) 
        self.previous_experience = None

    def collect_experience(self,
                           policy: Policy,
                           num_steps: int = 1,
                           bootstrap: bool = True,
                           ) -> Dict[str, torch.Tensor]:
        """
        Collects the given number of experiences from the environment using the provided policy.

        The experiences are collected across all environments, so the actual number of steps is ceil(num_steps / N) where N is the number of environments. The output shape is (T, N, ...) if not flattened, or (N*T, ...) if flattened. 
        
        Args:
            policy (Policy): A policy that implements the Policy interface.
            num_steps (int): The number of steps to collect experience for. Defaults to 1.
            bootstrap (bool): Whether to compute the last value estimate V(s_{T+1}) for bootstrapping if the last step is not done and the policy provides value estimates. Defaults to True.
            
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
                - 'state': The states collected. Shape (T, N, ...), or (N*T, ...) if flattened.
                - 'action': The actions taken. Shape (T, N, ...), or (N*T, ...) if flattened.
                - 'next_state': The next states after taking the actions. Shape (T, N, ...), or (N*T, ...) if flattened.
                - 'reward': The rewards received. Shape (T, N, 1), or (N*T, 1) if flattened.
                - 'done': The done flags indicating if the episode has ended. Shape (T, N, 1), or (N*T, 1) if flattened.
                - All keys from the policy's info dictionary (e.g., 'value', 'log_prob', etc.)
                - 'last_value_est' (optional): The last value estimate for bootstrapping, if applicable. (N, 1)
        """
        num_envs = self.env.get_num_envs()
        num_timesteps = (num_steps + num_envs - 1) // num_envs  # Ceiling division

        # Collect steps from all environments
        step_data = self._collect_steps(policy, num_timesteps)

        # Combine step data into tensors
        experience = self._combine_step_data(step_data, flatten=self.flatten)

        # Add bootstrap value estimate if applicable
        if bootstrap and 'value' in experience:
            _, policy_info = policy.act(self.previous_experience['next_state'])
            last_value_estimate = policy_info.get("value")
            if last_value_estimate is not None:
                experience['last_value_est'] = last_value_estimate

        return experience
    
    def collect_trajectory(self, 
                           policy: Policy,
                           num_trajectories: int | None = None,
                           min_num_steps: int | None = None,
                           ) -> Dict[str, torch.Tensor]:
        """
        Collects full trajectories in parallel from the environment using the provided policy.

        If the number of trajectories specified matches the number of environments, it will collect one trajectory from each environment. 
        If the number of trajectories is less than the number of environments, it will collect the specified number of trajectories from the first N environments. 
        If the number of trajectories is greater than the number of environments, it will collect num_trajectories // N trajectories from each environment, where N is the number of environments, 
        and then get the remaining trajectories from whichever environments complete first. 

        The output is a dictionary with keys (state, action, next_state, reward, done) where each key contains a tensor with the first dimension (B, ...) where B is the sum of each trajectories timesteps T.

        Args:
            policy (Policy | None): The policy or agent to use.
            num_trajectories (int | None): The total number of complete trajectories to collect.
            min_num_steps (int | None): The minimum number of steps to collect before completing the trajectories. If specified, will collect until the minimum number of steps is reached, then complete the last trajectory.
            
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
                - state: The current state of the environment. Shape (B, state_dim)
                - action: The action taken by the policy. Shape (B, action_dim)
                - next_state: The next state after taking the action. Shape (B, state_dim)
                - reward: The reward received from the environment. Shape (B, 1)
                - done: The done flag indicating if the episode has ended. Shape (B, 1)
                - All keys from the policy's info dictionary (e.g., 'value', 'log_prob', etc.)
        """
        # Validate and set default arguments
        if num_trajectories is None and min_num_steps is None:
            num_trajectories = 1
        if num_trajectories is not None and min_num_steps is not None:
            raise ValueError("Only one of num_trajectories or min_num_steps should be provided, not both.")

        num_envs = self.env.get_num_envs()

        # Force fresh full episodes by resetting previous experience
        self.previous_experience = None

        # Collect steps until stopping condition is met
        step_data, episode_tracking = self._collect_steps_until_episodes_complete(
            policy, num_envs, num_trajectories, min_num_steps
        )

        # Stack collected data into tensors with shape (T, N, ...)
        stacked_data = self._stack_step_data(step_data)

        # Find episode segments and select which ones to include
        episode_segments = self._find_episode_segments(stacked_data['done'], num_envs)
        
        if not episode_segments:
            raise ValueError("No complete episodes were recorded. Ensure the environment is properly configured and the policy is valid.")

        selected_segments = self._select_trajectory_segments(
            episode_segments, num_envs, num_trajectories, min_num_steps
        )

        # Extract and concatenate the selected trajectory segments
        return self._extract_trajectory_segments(stacked_data, selected_segments)

    def get_metric_tracker(self) -> MetricsTracker:
        """
        Returns the internal MetricsTracker instance for accessing collection metrics.

        Returns:
            MetricsTracker: The internal MetricsTracker instance.
        """
        return self.metric  

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _collect_steps(self, policy: Policy, num_timesteps: int) -> Dict[str, List[torch.Tensor]]:
        """
        Collects a fixed number of steps from all environments.

        Args:
            policy: The policy to use for action selection.
            num_timesteps: The number of timesteps to collect.

        Returns:
            Dictionary with lists of tensors for each data field.
        """
        step_data = {
            'state': [],
            'action': [],
            'next_state': [],
            'reward': [],
            'done': [],
            'policy_vals': {}
        }

        for _ in range(num_timesteps):
            state, action, next_state, reward, done, policy_vals = self._collect_step(policy)

            step_data['state'].append(state)
            step_data['action'].append(action)
            step_data['next_state'].append(next_state)
            step_data['reward'].append(reward)
            step_data['done'].append(done)
            
            self._accumulate_policy_vals(step_data['policy_vals'], policy_vals)

        return step_data

    def _collect_steps_until_episodes_complete(
        self,
        policy: Policy,
        num_envs: int,
        num_trajectories: int | None,
        min_num_steps: int | None
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, any]]:
        """
        Collects steps until the specified number of trajectories or minimum steps is reached.

        Args:
            policy: The policy to use for action selection.
            num_envs: The number of parallel environments.
            num_trajectories: Target number of complete trajectories (or None).
            min_num_steps: Minimum total steps from completed episodes (or None).

        Returns:
            Tuple of (step_data dict, episode_tracking dict).
        """
        step_data = {
            'state': [],
            'action': [],
            'next_state': [],
            'reward': [],
            'done': [],
            'policy_vals': {}
        }

        # Episode tracking state
        episodes_completed_per_env = [0] * num_envs
        current_episode_length = torch.zeros(num_envs, dtype=torch.long)
        total_completed_steps = 0

        while True:
            state, action, next_state, reward, done, policy_vals = self._collect_step(policy)

            step_data['state'].append(state)
            step_data['action'].append(action)
            step_data['next_state'].append(next_state)
            step_data['reward'].append(reward)
            step_data['done'].append(done)
            self._accumulate_policy_vals(step_data['policy_vals'], policy_vals)

            # Update episode tracking
            done_mask = done.view(-1).to(torch.bool)
            finished_env_indices = done_mask.nonzero(as_tuple=False).view(-1)
            
            current_episode_length += 1
            
            for env_index in finished_env_indices.tolist():
                episodes_completed_per_env[env_index] += 1
                total_completed_steps += int(current_episode_length[env_index].item())
                current_episode_length[env_index] = 0

            # Check stopping conditions
            if num_trajectories is not None:
                if self._have_enough_trajectories(num_trajectories, num_envs, episodes_completed_per_env):
                    break
            else:
                if total_completed_steps >= min_num_steps:
                    break

        if len(step_data['state']) == 0:
            raise ValueError("No steps were collected. Ensure the environment is properly configured and the policy is valid.")

        episode_tracking = {
            'episodes_per_env': episodes_completed_per_env,
            'total_completed_steps': total_completed_steps
        }

        return step_data, episode_tracking

    def _accumulate_policy_vals(
        self,
        accumulated: Dict[str, List[torch.Tensor]],
        new_vals: Dict[str, torch.Tensor]
    ) -> None:
        """
        Accumulates policy values into the accumulated dictionary.

        Args:
            accumulated: Dictionary to accumulate into.
            new_vals: New values to add.
        """
        for key, value in new_vals.items():
            if key not in accumulated:
                accumulated[key] = []
            accumulated[key].append(value)

    def _combine_step_data(
        self,
        step_data: Dict[str, List[torch.Tensor]],
        flatten: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Combines step data lists into tensors, either flattened or stacked.

        Args:
            step_data: Dictionary with lists of tensors.
            flatten: If True, concatenate along dim=0 for shape (N*T, ...).
                    If False, stack along dim=0 for shape (T, N, ...).

        Returns:
            Dictionary with combined tensors.
        """
        combine_fn = torch.cat if flatten else torch.stack
        dim = 0

        experience = {
            'state': combine_fn(step_data['state'], dim=dim),
            'action': combine_fn(step_data['action'], dim=dim),
            'next_state': combine_fn(step_data['next_state'], dim=dim),
            'reward': combine_fn(step_data['reward'], dim=dim),
            'done': combine_fn(step_data['done'], dim=dim),
        }

        # Add policy values
        for key, values in step_data['policy_vals'].items():
            experience[key] = combine_fn(values, dim=dim)

        return experience

    def _stack_step_data(self, step_data: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Stacks step data lists into tensors with shape (T, N, ...).

        Args:
            step_data: Dictionary with lists of tensors.

        Returns:
            Dictionary with stacked tensors.
        """
        stacked = {
            'state': torch.stack(step_data['state'], dim=0),
            'action': torch.stack(step_data['action'], dim=0),
            'next_state': torch.stack(step_data['next_state'], dim=0),
            'reward': torch.stack(step_data['reward'], dim=0),
            'done': torch.stack(step_data['done'], dim=0),
            'policy_vals': {
                key: torch.stack(values, dim=0)
                for key, values in step_data['policy_vals'].items()
            }
        }
        return stacked

    def _find_episode_segments(
        self,
        done_tensor: torch.Tensor,
        num_envs: int
    ) -> Dict[str, List[Tuple[int, int, int]]]:
        """
        Finds episode segments from the done tensor.

        An episode segment is defined by (env_index, start_timestep, end_timestep).

        Args:
            done_tensor: Tensor of done flags with shape (T, N, 1).
            num_envs: Number of environments.

        Returns:
            Dictionary with 'per_env' (list of lists) and 'global' (flat list) segments.
        """
        done_mask = done_tensor.squeeze(-1)  # Shape: (T, N)
        
        per_env_segments = [[] for _ in range(num_envs)]
        global_segments = []

        for env_index in range(num_envs):
            episode_end_timesteps = torch.nonzero(done_mask[:, env_index], as_tuple=False).flatten()
            
            if episode_end_timesteps.numel() == 0:
                continue

            previous_end = -1
            for end_timestep in episode_end_timesteps.tolist():
                start_timestep = previous_end + 1
                segment = (env_index, start_timestep, end_timestep)
                per_env_segments[env_index].append(segment)
                global_segments.append(segment)
                previous_end = end_timestep

        return {
            'per_env': per_env_segments,
            'global': global_segments
        }

    def _select_trajectory_segments(
        self,
        episode_segments: Dict[str, List],
        num_envs: int,
        num_trajectories: int | None,
        min_num_steps: int | None
    ) -> List[Tuple[int, int, int]]:
        """
        Selects which trajectory segments to include in the output.

        Args:
            episode_segments: Dictionary with 'per_env' and 'global' segment lists.
            num_envs: Number of environments.
            num_trajectories: Target number of trajectories (or None).
            min_num_steps: Minimum steps threshold (or None).

        Returns:
            List of selected segments (env_index, start, end).
        """
        per_env_segments = episode_segments['per_env']
        global_segments = episode_segments['global']

        if num_trajectories is not None:
            return self._select_segments_by_trajectory_count(
                per_env_segments, num_envs, num_trajectories
            )
        else:
            return self._select_segments_by_step_count(
                global_segments, min_num_steps
            )

    def _select_segments_by_trajectory_count(
        self,
        per_env_segments: List[List[Tuple[int, int, int]]],
        num_envs: int,
        num_trajectories: int
    ) -> List[Tuple[int, int, int]]:
        """
        Selects segments to get exactly num_trajectories complete episodes.

        Uses fair distribution: base episodes per env + earliest extras for remainder.

        Args:
            per_env_segments: List of segment lists, one per environment.
            num_envs: Number of environments.
            num_trajectories: Target number of trajectories.

        Returns:
            List of selected segments sorted by end time.
        """
        selected = []
        base_per_env = num_trajectories // num_envs
        remainder = num_trajectories % num_envs

        # Take base episodes from each environment
        for env_index in range(num_envs):
            env_segments = per_env_segments[env_index]
            num_to_take = min(base_per_env, len(env_segments))
            selected.extend(env_segments[:num_to_take])

        # Select remainder from earliest extra episodes across environments
        if remainder > 0:
            extra_candidates = []
            for env_index in range(num_envs):
                env_segments = per_env_segments[env_index]
                if len(env_segments) > base_per_env:
                    extra_candidates.append(env_segments[base_per_env])
            
            # Sort by end time and take earliest
            extra_candidates.sort(key=lambda segment: segment[2])
            selected.extend(extra_candidates[:remainder])

        # Sort all selected by end time for temporal ordering
        selected.sort(key=lambda segment: segment[2])
        return selected

    def _select_segments_by_step_count(
        self,
        global_segments: List[Tuple[int, int, int]],
        min_num_steps: int
    ) -> List[Tuple[int, int, int]]:
        """
        Selects segments until total steps reaches min_num_steps.

        Takes episodes in order of completion time.

        Args:
            global_segments: List of all segments across environments.
            min_num_steps: Minimum total steps threshold.

        Returns:
            List of selected segments.
        """
        # Sort by end time (completion order)
        sorted_segments = sorted(global_segments, key=lambda segment: segment[2])
        
        selected = []
        accumulated_steps = 0
        
        for segment in sorted_segments:
            selected.append(segment)
            segment_length = segment[2] - segment[1] + 1
            accumulated_steps += segment_length
            
            if accumulated_steps >= min_num_steps:
                break

        return selected

    def _extract_trajectory_segments(
        self,
        stacked_data: Dict[str, torch.Tensor],
        selected_segments: List[Tuple[int, int, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts and concatenates selected trajectory segments into output tensors.

        Args:
            stacked_data: Dictionary with stacked tensors of shape (T, N, ...).
            selected_segments: List of (env_index, start, end) tuples.

        Returns:
            Dictionary with concatenated tensors of shape (B, ...).
        """
        # Initialize lists for each output field
        segment_tensors = {
            'state': [],
            'action': [],
            'next_state': [],
            'reward': [],
            'done': [],
        }
        
        policy_vals_tensors = {
            key: [] for key in stacked_data['policy_vals'].keys()
        }

        # Extract each segment
        for env_index, start_time, end_time in selected_segments:
            time_slice = slice(start_time, end_time + 1)
            
            segment_tensors['state'].append(stacked_data['state'][time_slice, env_index])
            segment_tensors['action'].append(stacked_data['action'][time_slice, env_index])
            segment_tensors['next_state'].append(stacked_data['next_state'][time_slice, env_index])
            segment_tensors['reward'].append(stacked_data['reward'][time_slice, env_index])
            segment_tensors['done'].append(stacked_data['done'][time_slice, env_index])
            
            for key, tensor in stacked_data['policy_vals'].items():
                policy_vals_tensors[key].append(tensor[time_slice, env_index])

        # Concatenate all segments
        if len(segment_tensors['state']) == 0:
            raise ValueError("No complete episodes were recorded. Ensure the environment is properly configured and the policy is valid.")

        output = {
            'state': torch.cat(segment_tensors['state'], dim=0),
            'action': torch.cat(segment_tensors['action'], dim=0),
            'next_state': torch.cat(segment_tensors['next_state'], dim=0),
            'reward': torch.cat(segment_tensors['reward'], dim=0),
            'done': torch.cat(segment_tensors['done'], dim=0),
        }

        for key, tensors in policy_vals_tensors.items():
            if tensors:
                output[key] = torch.cat(tensors, dim=0)

        return output
    
    @staticmethod
    def _have_enough_trajectories(
        target_trajectories: int,
        num_envs: int,
        episodes_completed_per_env: List[int]
    ) -> bool:
        """
        Check if we have collected at least the target number of full trajectories.
        
        Args:
            target_trajectories: The target number of trajectories to collect.
            num_envs: The number of parallel environments.
            episodes_completed_per_env: List of completed episode counts per environment.
            
        Returns:
            True if we have collected enough trajectories, False otherwise.
        """
        base_per_env = target_trajectories // num_envs
        remainder = target_trajectories % num_envs

        # Check if all envs have at least the base number of episodes
        if any(count < base_per_env for count in episodes_completed_per_env):
            return False
        
        # If evenly divisible, we're done
        if remainder == 0:
            return True
        
        # Need at least `remainder` envs with one extra episode
        envs_with_extra = sum(1 for count in episodes_completed_per_env if count >= base_per_env + 1)
        return envs_with_extra >= remainder    

    def _collect_step(
        self,
        policy: Policy
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]: 
        """
        Collects a single step from the environment using the provided policy.
        
        Args:
            policy: A policy that implements the Policy interface.
        
        Returns:
            Tuple containing:
                - state: The current state. Shape (N, state_dim)
                - action: The action taken. Shape (N, action_dim)
                - next_state: The next state. Shape (N, state_dim)
                - reward: The reward received. Shape (N, 1)
                - done: The done flag. Shape (N, 1)
                - policy_vals: Dictionary of additional policy outputs.
        """
        # Get or reset state
        if self.previous_experience is None:
            state, _ = self.env.reset()
        else:
            state = self.previous_experience["next_state"].clone()
            # Reset individual environments that were done
            for env_index in range(self.previous_experience["done"].shape[0]):
                if self.previous_experience["done"][env_index]:
                    reset_state, _ = self.env.reset_index(env_index)
                    state[env_index] = reset_state

        # Get action from policy
        action, policy_vals = policy.act(state, deterministic=False)

        # Step environment
        next_state, reward, done, _ = self.env.step(action)

        # Update metrics
        self.metric.update(reward, done)

        # Save experience for next step
        self.previous_experience = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }

        return state, action, next_state, reward, done, policy_vals                
    
