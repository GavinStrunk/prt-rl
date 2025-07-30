import torch
from typing import Dict, Optional, List, Tuple
from prt_rl.env.interface import EnvironmentInterface, EnvParams, MultiAgentEnvParams
from prt_rl.common.loggers import Logger
from prt_rl.common.policies import ActorCriticPolicy, DistributionPolicy

def random_action(env_params: EnvParams, state: torch.Tensor) -> torch.Tensor:
    """
    Randomly samples an action from action space.
    Args:
        state (torch.Tensor): The current state of the environment.
    Returns:
        torch.Tensor: A tensor containing the sampled action.
    """
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
                               size=ashape)
    else:
        action = torch.rand(size=ashape)
        # Scale the random [0,1] actions to the action space [min,max]
        max_actions = torch.tensor(params.action_max).unsqueeze(0)
        min_actions = torch.tensor(params.action_min).unsqueeze(0)
        action = action * (max_actions - min_actions) + min_actions
    return action 

def get_action_from_policy(policy, state: torch.Tensor, env_params: EnvParams = None) -> torch.Tensor:
    """
    Get an action from the policy given the state.
    
    Args:
        policy: The policy to get the action from.
        state (torch.Tensor): The current state of the environment.
    
    Returns:
        torch.Tensor: The action to take.
    """
    # Ensure the state is float32
    state = state.float()

    value_estimate = None
    log_prob = None 

    if policy is None:
        if env_params is not None:
            action = random_action(env_params, state)
        else:
            raise ValueError("env_params must be provided if policy is None")
    elif isinstance(policy, ActorCriticPolicy):
        action, value_estimate, log_prob = policy.predict(state)
    elif isinstance(policy, DistributionPolicy):
        action, log_prob = policy.predict(state)
    else:
        action = policy(state)
    
    return action, value_estimate, log_prob

class SequentialCollector:
    """
    The Sequential Collector collects experience from a single environment sequentially.
    It resets the environment when the previous experience is done.

    Args:
        env (EnvironmentInterface): The environment to collect experience from.
        logger (Optional[Logger]): Optional logger for logging information. Defaults to a new Logger instance.
        logging_freq (int): Frequency of logging experience collection. Defaults to 1.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 logger: Optional[Logger] = None,
                 logging_freq: int = 1,
                 seed: Optional[int] = None
                 ) -> None:
        self.env = env
        self.env_params = env.get_parameters()
        self.logger = logger or Logger.create('blank')
        self.logging_freq = logging_freq
        self.seed = seed
        self.previous_experience = None
        self.collected_steps = 0
        self.previous_episode_reward = 0
        self.previous_episode_length = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.cumulative_reward = 0
        self.num_episodes = 0

    def _random_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Randomly samples an action from action space.

        Args:
            state (torch.Tensor): The current state of the environment.
        Returns:
            torch.Tensor: A tensor containing the sampled action.
        """
        if isinstance(self.env_params, EnvParams):
            ashape = (state.shape[0], self.env_params.action_len)
            params = self.env_params
        elif isinstance(self.env_params, MultiAgentEnvParams):
            ashape = (state.shape[0], self.env_params.num_agents, self.env_params.agent.action_len)
            params = self.env_params.agent
        else:
            raise ValueError("env_params must be a EnvParams or MultiAgentEnvParams")

        if not params.action_continuous:
            # Add 1 to the high value because randint samples between low and 1 less than the high: [low,high)
            action = torch.randint(low=params.action_min, high=params.action_max + 1,
                                   size=ashape)
        else:
            action = torch.rand(size=ashape)

            # Scale the random [0,1] actions to the action space [min,max]
            max_actions = torch.tensor(params.action_max).unsqueeze(0)
            min_actions = torch.tensor(params.action_min).unsqueeze(0)
            action = action * (max_actions - min_actions) + min_actions

        return action 

    def collect_experience(self,
                           policy = None,
                           num_steps: int = 1
                           ) -> Dict[str, torch.Tensor]:
        """
        Collects the given number of experiences from the environment using the provided policy. Returns experience with shape (T, ...).
        Args:
            policy (callable): A callable that takes a state and returns an action.
            num_steps (int): The number of steps to collect experience for. Defaults to 1.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
                - 'state': The states collected.
                - 'action': The actions taken.
                - 'next_state': The next states after taking the actions.
                - 'reward': The rewards received.
                - 'done': The done flags indicating if the episode has ended.
        """
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        for _ in range(num_steps):
            # Reset the environment if no previous state
            if self.previous_experience is None or self.previous_experience["done"]:
                state, _ = self.env.reset(seed=self.seed)
                self.seed = self.seed + 1 if self.seed is not None else None
            else:
                state = self.previous_experience["next_state"]

            action, value_est, log_prob = get_action_from_policy(policy, state, self.env_params)

            next_state, reward, done, _ = self.env.step(action)

            states.append(state.squeeze(0))
            actions.append(action.squeeze(0))
            next_states.append(next_state.squeeze(0))
            rewards.append(reward.squeeze(0))
            dones.append(done.squeeze(0))

            self.collected_steps += 1
            self.current_episode_reward += reward.sum().item()
            self.current_episode_length += 1
            self.cumulative_reward += reward.sum().item()

            if done:
                self.previous_episode_reward = self.current_episode_reward
                self.previous_episode_length = self.current_episode_length
                self.current_episode_reward = 0
                self.current_episode_length = 0
                self.num_episodes += 1

            self.previous_experience = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }

        if self.collected_steps % self.logging_freq == 0:
            self.logger.log_scalar(name='episode_reward', value=self.previous_episode_reward, iteration=self.collected_steps)
            self.logger.log_scalar(name='episode_length', value=self.previous_episode_length, iteration=self.collected_steps)
            self.logger.log_scalar(name='cumulative_reward', value=self.cumulative_reward, iteration=self.collected_steps)
            self.logger.log_scalar(name='episode_number', value=self.num_episodes, iteration=self.collected_steps)

        return {
            "state": torch.stack(states, dim=0),
            "action": torch.stack(actions, dim=0),
            "next_state": torch.stack(next_states, dim=0),
            "reward": torch.stack(rewards, dim=0),
            "done": torch.stack(dones, dim=0),
        }
    def collect_trajectory(self, 
                           policy = None,
                           num_trajectories: Optional[int] = None,
                           min_num_steps: Optional[int] = None
                           ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Collects a single trajectory from the environment using the provided policy.

        Return the trajectory with the shape (T, ...), where T is the number of steps in the trajectory.
        Args:
            policy (callable): A callable that takes a state and returns an action.
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

    def _collect_single_trajectory(self, policy) -> Dict[str, torch.Tensor]:
        """
        Collects a single trajectory from the environment using the provided policy.

        Returns a trajectory with the shape (T, ...), where T is the number of steps in the trajectory.
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

            self.collected_steps += 1
            self.current_episode_reward += reward.sum().item()
            self.current_episode_length += 1
            self.cumulative_reward += reward.sum().item()

            if done:
                self.previous_episode_reward = self.current_episode_reward
                self.previous_episode_length = self.current_episode_length
                self.current_episode_reward = 0
                self.current_episode_length = 0
                self.num_episodes += 1

                if self.num_episodes % self.logging_freq == 0:
                    self.logger.log_scalar(name='episode_reward', value=self.previous_episode_reward, iteration=self.collected_steps)
                    self.logger.log_scalar(name='episode_length', value=self.previous_episode_length, iteration=self.collected_steps)
                    self.logger.log_scalar(name='cumulative_reward', value=self.cumulative_reward, iteration=self.collected_steps)
                    self.logger.log_scalar(name='episode_number', value=self.num_episodes, iteration=self.collected_steps)
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
        logging_freq (int): Frequency of logging experience collection. Defaults to 1.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 flatten: bool = True,
                 logger: Optional[Logger] = None,
                 logging_freq: int = 1
                 ) -> None:
        self.env = env
        self.env_params = env.get_parameters()
        self.flatten = flatten
        self.logger = logger or Logger.create('blank')
        self.logging_freq = logging_freq
        self.previous_experience = None
        self.collected_steps = 0
        self.previous_episode_reward = 0
        self.previous_episode_length = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.cumulative_reward = 0
        self.num_episodes = 0

    # def _random_action(self, state: torch.Tensor) -> torch.Tensor:
    #     """
    #     Randomly samples an action from action space.

    #     Args:
    #         state (torch.Tensor): The current state of the environment.
    #     Returns:
    #         torch.Tensor: A tensor containing the sampled action.
    #     """
    #     if isinstance(self.env_params, EnvParams):
    #         ashape = (state.shape[0], self.env_params.action_len)
    #         params = self.env_params
    #     elif isinstance(self.env_params, MultiAgentEnvParams):
    #         ashape = (state.shape[0], self.env_params.num_agents, self.env_params.agent.action_len)
    #         params = self.env_params.agent
    #     else:
    #         raise ValueError("env_params must be a EnvParams or MultiAgentEnvParams")

    #     if not params.action_continuous:
    #         # Add 1 to the high value because randint samples between low and 1 less than the high: [low,high)
    #         action = torch.randint(low=params.action_min, high=params.action_max + 1,
    #                                size=ashape)
    #     else:
    #         action = torch.rand(size=ashape)

    #         # Scale the random [0,1] actions to the action space [min,max]
    #         max_actions = torch.tensor(params.action_max).unsqueeze(0)
    #         min_actions = torch.tensor(params.action_min).unsqueeze(0)
    #         action = action * (max_actions - min_actions) + min_actions

    #     return action 

    def collect_experience(self,
                           policy = None,
                           num_steps: int = 1
                           ) -> Dict[str, torch.Tensor]:
        """
        Collects the given number of experiences from the environment using the provided policy.
        Args:
            policy (callable): A callable that takes a state and returns an action.
            num_steps (int): The number of steps to collect experience for. Defaults to 1.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
                - 'state': The states collected.
                - 'action': The actions taken.
                - 'next_state': The next states after taking the actions.
                - 'reward': The rewards received.
                - 'done': The done flags indicating if the episode has ended.
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
            if value_estimates:
                value_estimates = torch.cat(value_estimates, dim=0)
            if log_probs:
                log_probs = torch.cat(log_probs, dim=0)
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
        if log_probs is not None:
            experience['log_prob'] = log_probs

        # Compute the last value estimate for boostrapping
        if isinstance(policy, ActorCriticPolicy):
            # Compute the last value estimate
            _, last_value_estimate, _ = policy.predict(self.previous_experience['next_state'])
            experience['last_value_est'] = last_value_estimate
        
        return experience
    
    def collect_trajectory(self, 
                           policy = None,
                           num_trajectories: Optional[int] = None,
                           min_num_steps: Optional[int] = None
                           ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Collects a single trajectory from the environment using the provided policy.

        Return the trajectory with the shape (T, ...), where T is the number of steps in the trajectory.
        Args:
            policy (callable): A callable that takes a state and returns an action.
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
        
        N = self.env.get_num_envs()
        trajectories = []
        total_steps = 0

        return trajectories, total_steps    