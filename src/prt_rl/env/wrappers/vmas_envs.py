"""
Vectorized Multi-Agent Simulator (VMAS) Environment Wrapper
"""
from collections import Counter
import torch
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
import vmas
from prt_rl.env.interface import MultiAgentEnvironmentInterface, MultiAgentEnvParams, EnvParams, MultiGroupEnvironmentInterface, MultiGroupEnvParams
from prt_rl.env.wrappers.gymnasium_envs import GymnasiumWrapper


class VmasWrapper(MultiAgentEnvironmentInterface):
    """
    Vectorized Multi-Agent Simulator (VMAS)

    The VMAS wrapper provides an interface to VMAS multi-agent environments where all agents belong to a single group. VmasMultiGroupWrapper should be used for environments with multiple agent groups.

    Examples:
        .. code-block:: python
            from prt_rl.env.wrappers import VmasWrapper

            env = VmasWrapper(
                scenario="discovery",
                num_envs=4,
            )

    Args:
        scenario (str): Name of the VMAS environment
        render_mode (str): Render mode for the environment. Options are None or 'rgb_array'.

    References:
        [1] https://github.com/proroklab/VectorizedMultiAgentSimulator
    """

    def __init__(self,
                 scenario: str,
                 render_mode: Optional[str] = None,
                 **kwargs
                 ) -> None:
        super().__init__(render_mode)
        self.env = vmas.make_env(
            scenario,
            **kwargs,
        )
        self.env_params = self._make_env_params()

    def get_parameters(self) -> MultiAgentEnvParams:
        return self.env_params

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Resets the environment to the initial state and returns the initial observation.

        Args:
            seed (int | None): Sets the random seed.

        Returns:
            Tuple: Tuple of tensors containing the initial observation and info dictionary
        """
        info = {}
        state = self.env.reset(seed=seed)

        # Stack the observation so it has shape (# env, # agents, obs shape)
        state = torch.stack(state, dim=1)

        if self.render_mode == 'rgb_array':
            rgb = self.env.render(mode=self.render_mode)

            # Fix the negative stride in the numpy array
            img = rgb.copy()
            info['rgb_array'] = torch.from_numpy(img).unsqueeze(0)

        return state, info

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Steps the simulation using the action tensor and returns the new trajectory.

        Args:
            action (torch.Tensor): Tensor with "action" key that is a tensor with shape (# env, # agents, # actions)

        Returns:
            Tuple: Tuple of tensors containing the next state, reward, done, and info dictionary
        """
        # VMAS expects actions to have shape (# agents, # env, action shape)
        action_val = action.permute(1, 0, 2)

        next_state, reward, done, info = self.env.step(action_val)
        next_state = torch.stack(next_state, dim=1)
        reward = torch.stack(reward, dim=1)
        done = done.unsqueeze(-1)

        if self.render_mode == 'rgb_array':
            rgb = self.env.render(mode=self.render_mode)

            # Fix the negative stride in the numpy array
            img = rgb.copy()
            info['rgb_array'] = torch.from_numpy(img).unsqueeze(0)

        return next_state, reward, done, info
    
    def close(self) -> None:
        """
        Closes the environment and cleans up any resources.
        """
        return self.env.close()

    def _make_env_params(self):
        # Get the agent names
        agent_names = [a.name for a in self.env.agents]

        # Extract group names by matching prefixes with the pattern 'agent_0', 'agent_1' and count the agents with the same prefix
        name_prefixes = Counter(item.rsplit('_', 1)[0] for item in agent_names)

        # Convert to a list of lists containing [[group_name, agent_count],[...]]
        group_list = [[key, count] for key, count in name_prefixes.items()]

        # If there is more than one group this is not a MultiAgent environment
        if len(group_list) > 1:
            raise ValueError("VmasWrapper only supports single group multi-agent environments.")

        # For each group create a MultiAgentEnvParams object
        group = {}
        agent_index = 0
        for name, count in group_list:
            # Construct the EnvParams for an agent in the group
            action_space = self.env.action_space[agent_index]
            # It appears the gymnasium and gym spaces do not pass isinstance
            act_shape, act_cont, act_min, act_max = GymnasiumWrapper._get_params_from_box(action_space)
            if len(act_shape) == 1:
                action_len = act_shape[0]
            else:
                raise ValueError(f"Action space does not have 1D shape: {act_shape}")

            observe_space = self.env.observation_space[agent_index]
            obs_shape, obs_cont, obs_min, obs_max = GymnasiumWrapper._get_params_from_box(observe_space)

            agent_params = EnvParams(
                action_len=action_len,
                action_min=act_min,
                action_max=act_max,
                action_continuous=self.env.continuous_actions,
                observation_shape=obs_shape,
                observation_continuous=obs_cont,
                observation_min=obs_min,
                observation_max=obs_max,
            )

            # Construct a MultiAgentEnvParams consisting of the number of agents in this group
            ma_params = MultiAgentEnvParams(
                num_agents=count,
                agent=agent_params
            )
            group[name] = ma_params

            # The action and observation space are a flat list with values for each agent so we need to index the next group of agents
            agent_index += count

        return group[list(group.keys())[0]]

class VmasMultiGroupWrapper(MultiGroupEnvironmentInterface):
    """
    Vectorized Multi-Agent Simulator (VMAS) Multi-Group Environment Wrapper

    The VMAS Multi-Group wrapper provides an interface to VMAS multi-agent environments where agents belong to multiple groups. This wrapper implements the MultiGroupEnvironmentInterface.
    
    Examples:
        .. code-block:: python
            from prt_rl.env.wrappers import VmasMultiGroupWrapper

            env = VmasMultiGroupWrapper(
                scenario="kinematic_bicycle",
                num_envs=4,
            )

    Args:
        scenario (str): Name of the VMAS environment
        render_mode (str): Render mode for the environment. Options are None or 'rgb_array'.

    References:
        [1] https://github.com/proroklab/VectorizedMultiAgentSimulator
    """
    def __init__(self,
                 scenario: str,
                 render_mode: Optional[str] = None,
                 **kwargs
                 ) -> None:
        super().__init__(render_mode)
        self.env = vmas.make_env(
            scenario,
            **kwargs,
        )
        self.env_params = self._make_env_params()

    def get_parameters(self) -> MultiGroupEnvParams:
        return self.env_params
    
    def reset(self, seed: int | None = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Resets the environment to the initial state and returns the initial observation.

        Args:
            seed (int | None): Sets the random seed.

        Returns:
            Tuple: Tuple of tensors containing the initial observation and info dictionary
        """
        info = {}
        state = {}

        # Returns a list of tensors with shape (# env, obs shape)
        raw_state = self.env.reset(seed=seed)

        for i, group in enumerate(self.group_list):
            group_name, _ = group
            # Ensure observation has shape (# env, # agents, obs shape)
            if raw_state[i].ndim == 2:
                raw_state[i] = raw_state[i].unsqueeze(1)

            state[group_name] = raw_state[i]

        if self.render_mode == 'rgb_array':
            rgb = self.env.render(mode=self.render_mode)

            # Fix the negative stride in the numpy array
            img = rgb.copy()
            info['rgb_array'] = torch.from_numpy(img).unsqueeze(0)

        return state, info

    def step(self, action: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Steps the simulation using the action tensor and returns the new trajectory.

        Args:
            action (torch.Tensor): Tensor with "action" key that is a tensor with shape (# env, # agents, # actions)

        Returns:
            Tuple: Tuple of tensors containing the next state, reward, done, and info dictionary
        """
        # VMAS expects actions to have shape (# agents, # env, action shape)
        actions = []
        for group in self.group_list:
            group_name, _ = group

            # Convert the actions for the group from (# env, # agents, action shape) to (# agents, # env, action shape)
            group_actions = action[group_name].permute(1, 0, 2)

            # Convert the tensor to a list with length # agents where each entry has shape (# env, action shape)
            action_list = list(torch.unbind(group_actions, dim=0))

            # Create a flat list of actions for all agents
            actions.extend(action_list)

        next_state, reward, done, info = self.env.step(actions)

        next_states = {}
        rewards = {}
        for i, group in enumerate(self.group_list):
            group_name, _ = group
            # Ensure observation has shape (# env, # agents, obs shape)
            if next_state[i].ndim == 2:
                next_state[i] = next_state[i].unsqueeze(1)

            next_states[group_name] = next_state[i] 
            rewards[group_name] = reward[i].unsqueeze(-1)  

        done = done.unsqueeze(-1)

        if self.render_mode == 'rgb_array':
            rgb = self.env.render(mode=self.render_mode)

            # Fix the negative stride in the numpy array
            img = rgb.copy()
            info['rgb_array'] = torch.from_numpy(img).unsqueeze(0)

        return next_states, rewards, done, info
    
    def close(self) -> None:
        """
        Closes the environment and cleans up any resources.
        """
        return self.env.close()

    def _make_env_params(self):
        # Get the agent names
        agent_names = [a.name for a in self.env.agents]

        # Extract group names by matching prefixes with the pattern 'agent_0', 'agent_1' and count the agents with the same prefix
        name_prefixes = Counter(item.rsplit('_', 1)[0] for item in agent_names)

        # Convert to a list of lists containing [[group_name, agent_count],[...]]
        self.group_list = [[key, count] for key, count in name_prefixes.items()]

        # For each group create a MultiAgentEnvParams object
        group = {}
        agent_index = 0
        for name, count in self.group_list:
            # Construct the EnvParams for an agent in the group
            action_space = self.env.action_space[agent_index]
            # It appears the gymnasium and gym spaces do not pass isinstance
            act_shape, act_cont, act_min, act_max = GymnasiumWrapper._get_params_from_box(action_space)
            if len(act_shape) == 1:
                action_len = act_shape[0]
            else:
                raise ValueError(f"Action space does not have 1D shape: {act_shape}")

            observe_space = self.env.observation_space[agent_index]
            obs_shape, obs_cont, obs_min, obs_max = GymnasiumWrapper._get_params_from_box(observe_space)

            agent_params = EnvParams(
                action_len=action_len,
                action_min=act_min,
                action_max=act_max,
                action_continuous=self.env.continuous_actions,
                observation_shape=obs_shape,
                observation_continuous=obs_cont,
                observation_min=obs_min,
                observation_max=obs_max,
            )

            # Construct a MultiAgentEnvParams consisting of the number of agents in this group
            ma_params = MultiAgentEnvParams(
                num_agents=count,
                agent=agent_params
            )
            group[name] = ma_params

            # The action and observation space are a flat list with values for each agent so we need to index the next group of agents
            agent_index += count

        return MultiGroupEnvParams(group=group)  