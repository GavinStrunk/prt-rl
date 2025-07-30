from collections import Counter
import gymnasium as gym
import numpy as np
import prt_sim.jhu
import prt_sim.jhu.bandits
import prt_sim.jhu.base
from tensordict.tensordict import TensorDict
import torch
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
import vmas
import prt_sim
from prt_rl.env.interface import EnvironmentInterface, EnvParams, MultiAgentEnvParams, MultiGroupEnvParams, NumpyEnvironmentInterface


class JhuWrapper(NumpyEnvironmentInterface):
    """
    Wraps the JHU environments in the Environment interface.

    The JHU environments are games and puzzles that were used in the JHU 705.741 RL course.

    Args:
        environment (BaseEnvironment): JHU Environment object
        render_mode (str, optional): Sets the render mode ['human', 'rgb_array']. Default: None.

    Examples:
        ```python
        from prt_sim.jhu.bandits import KArmBandits
        from prt_rl.env.wrappers import JhuWrapper
        from prt_rl.common.policy import RandomPolicy

        env = JhuWrapper(environment=KArmBandits())
        policy = RandomPolicy(env_params=env.get_parameters())

        state = env.reset(seed=0)
        done = False

        while not done:
            action = policy.get_action(state)
            next_state, reward, done, info = env.step(action)

        ```
    """

    def __init__(self,
                 environment: prt_sim.jhu.base.BaseEnvironment,
                 render_mode: Optional[str] = None,
                 ) -> None:
        super().__init__(render_mode)
        self.env = environment

    def get_parameters(self) -> EnvParams:
        """
        Returns the EnvParams object which contains information about the sizes of observations and actions needed for setting up RL agents.
        Returns:
            EnvParams: environment parameters object
        """
        params = EnvParams(
            action_len=1,
            action_continuous=False,
            action_min=0,
            action_max=self.env.get_number_of_actions() - 1,
            observation_shape=(1,),
            observation_continuous=False,
            observation_min=0,
            observation_max=max(self.env.get_number_of_states() - 1, 0),
        )
        return params

    def reset(self, seed: int | None = None) -> np.ndarray:
        """
        Resets the environment to the initial state and returns the initial observation.
        Args:
            seed (int | None): Sets the random seed.
        Returns:
            Tuple: Tuple of numpy arrays containing the initial observation and info dictionary
        """
        info = {}
        state = self.env.reset(seed=seed)
        state = np.array([[state]], dtype=np.int64)

        # Add info for Bandit environment
        if isinstance(self.env, prt_sim.jhu.bandits.KArmBandits):
            info = {
                'optimal_bandit': np.array([[self.env.get_optimal_bandit()]], dtype=np.int64),
                'bandits': np.array([self.env.bandit_probs])
            }

        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            rgb = self.env.render()
            info['rgb_array'] = rgb[np.newaxis, ...]

        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Steps the simulation using the action tensor and returns the new trajectory.
        Args:
            action (np.ndarray): Numpy array with shape (# env, # actions)
        Returns:
            Tuple: Tuple of numpy arrays containing the next state, reward, done, and info dictionary
        """
        info = {}
        state, reward, done = self.env.execute_action(action[0][0])

        # Convert integers to numpy arrays
        state = np.array([[state]], dtype=np.int64)
        reward = np.array([[reward]], dtype=np.float32)
        done = np.array([[done]], dtype=np.bool)

        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            rgb = self.env.render()
            info['rgb_array'] = rgb[np.newaxis, ...]

        return state, reward, done, info


class GymnasiumWrapper(EnvironmentInterface):
    """
    Wraps the Gymnasium environments in the Environment interface.

    Args:
        gym_name: Name of the Gymnasium environment.
        num_envs: Number of parallel environments to create.
        render_mode: Sets the rendering mode. Defaults to None.

    Examples:
        ```python
        from prt_rl.env.wrappers import GymnasiumWrapper
        from prt_rl.common.policy import RandomPolicy

        env = GymnasiumWrapper(
            gym_name="CarRacing-v3",
            render_mode="rgb_array",
            continuous=True
        )

        policy = RandomPolicy(env_params=env.get_parameters())

        state, info = env.reset()
        done = False

        while not done:
            action = policy.get_action(state)
            next_state, reward, done, info = env.step(action)

    """

    def __init__(self,
                 gym_name: str,
                 num_envs: int = 1,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None,
                 device: str = 'cpu',
                 **kwargs
                 ) -> None:
        super().__init__(render_mode, num_envs=num_envs)
        self.gym_name = gym_name
        self.device = torch.device(device)

        if self.num_envs == 1:
            self.env = gym.make(self.gym_name, render_mode=render_mode, **kwargs)

            # Seed the environment if a seed is provided
            if seed is not None:
                self.env.reset(seed=seed)
                self.env.action_space.seed(seed)
                self.env.observation_space.seed(seed)
            vectorized = False
        else:
            def make_env_fn(env_index: int):
                def _init():
                    env = gym.make(gym_name, render_mode=render_mode, **kwargs)
                    
                    # Seed the environment if a seed is provided
                    if seed is not None:
                        env_seed = seed + env_index
                        env.reset(seed=env_seed)
                        env.action_space.seed(env_seed)
                        env.observation_space.seed(env_seed)
                    return env
                return _init

            self.env = gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
            vectorized = True

        self.env_params = self._make_env_params(vectorized=vectorized)

    def get_parameters(self) -> EnvParams:
        """
        Returns the EnvParams object which contains information about the sizes of observations and actions needed for setting up RL agents.
        Returns:
            EnvParams: environment parameters object
        """
        return self.env_params

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Resets the environment to the initial state and returns the initial observation.
        Args:
            seed (int | None): Sets the random seed.
        Returns:
            Tuple: Tuple of tensors containing the initial observation and info dictionary
        """
        state, info = self.env.reset(seed=seed)
        state = self._process_observation(state)

        if self.render_mode == 'rgb_array':
            rgb = self.env.render()
            info['rgb_array'] = rgb[np.newaxis, ...]
            
        return state, info
    
    def reset_index(self, index: int, seed: int | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Resets only the environments that are done.

        Args:
            done (torch.Tensor): Boolean tensor of shape (num_envs, 1) or (num_envs,)

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The new observations and info dict
        """
        if index > self.num_envs:
            raise ValueError(f"Index {index} is out of bounds for {self.num_envs} environments.")
        
        state, info = self.env.envs[index].reset(seed=seed)
        state = self._process_observation(state)

        if self.render_mode == 'rgb_array':
            rgb = self.env.render()
            info['rgb_array'] = rgb[np.newaxis, ...]

        return state, info

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Steps the simulation using the action tensor and returns the new trajectory.
        Args:
            action (torch.Tensor): Tensor with "action" key that is a tensor with shape (# env, # actions)
        Returns:
            Tuple: Tuple of tensors containing the next state, reward, done, and info dictionary
        """
        # Discrete actions send the raw integer value to the step function
        if not self.env_params.action_continuous:
            if self.num_envs == 1:
                # If there is only one environment, the step function expects a single integer action
                action = action.item()
            else:
                # If there are multiple environments and 1 action, the step function expects an action with shape (# envs,)
                action = action.cpu().numpy().squeeze(-1)
        else:
            action = action.cpu().numpy()

            # If there is only one environment remove the first dimension
            if action.shape[0] == 1:
                action = action[0]

        next_state, reward, terminated, trunc, info = self.env.step(action)
        done = np.logical_or(terminated, trunc)

        # Reshape the reward and done to be (# envs, 1)
        if self.num_envs == 1:
            reward = torch.tensor([[reward]], dtype=torch.float, device=self.device)
            done = torch.tensor([[done]], dtype=torch.bool, device=self.device)
        else:
            reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(-1)
            done = torch.tensor(done, dtype=torch.bool, device=self.device).unsqueeze(-1)

        next_state = self._process_observation(next_state)

        if self.render_mode == 'rgb_array':
            rgb = self.env.render()
            info['rgb_array'] = rgb[np.newaxis, ...]

        return next_state, reward, done, info
    
    def close(self):
        return self.env.close()

    def _process_observation(self, observation: Union[torch.Tensor | int]) -> torch.Tensor:
        """
        Processes the observation to ensure it is in the correct format.
        Args:
            observation (Union[torch.Tensor | int]): The observation to process.
        Returns:
            torch.Tensor: The processed observation.
        """
        if isinstance(observation, int):
            observation = np.array([observation])

        # Add a dimension if there is only 1 environment
        if self.num_envs == 1:
            observation = torch.tensor(observation, device=self.device).unsqueeze(0)
        else:
            observation = torch.tensor(observation, device=self.device)

        return observation

    def _make_env_params(self,
                         vectorized: bool = False,
                         ) -> EnvParams:
        """
        Creates the environment parameters based on the action and observation space of the environment.
        Args:
            vectorized (bool): If True, the environment is vectorized.
        Returns:
            EnvParams: The environment parameters object.
        """
        if not vectorized:
            action_space = self.env.action_space
            observation_space = self.env.observation_space
        else:
            action_space = self.env.single_action_space
            observation_space = self.env.single_observation_space

        if isinstance(action_space, gym.spaces.Discrete):
            act_shape, act_cont, act_min, act_max = self._get_params_from_discrete(action_space)
            action_len = act_shape[0]
        elif isinstance(action_space, gym.spaces.Box):
            act_shape, act_cont, act_min, act_max = self._get_params_from_box(action_space)
            if len(act_shape) == 1:
                action_len = act_shape[0]
            else:
                raise ValueError(f"Action space does not have 1D shape: {act_shape}")
        else:
            raise NotImplementedError(f"{action_space} action space is not supported")

        if isinstance(observation_space, gym.spaces.Discrete):
            obs_shape, obs_cont, obs_min, obs_max = self._get_params_from_discrete(observation_space)
        elif isinstance(observation_space, gym.spaces.Box):
            obs_shape, obs_cont, obs_min, obs_max = self._get_params_from_box(observation_space)
        else:
            raise NotImplementedError(f"{observation_space} observation space is not supported")

        return EnvParams(
            action_len=action_len,
            action_continuous=act_cont,
            action_min=act_min,
            action_max=act_max,
            observation_shape=obs_shape,
            observation_continuous=obs_cont,
            observation_min=obs_min,
            observation_max=obs_max,
        )

    @staticmethod
    def _get_params_from_discrete(space: gym.spaces.Discrete) -> Tuple[tuple, bool, int, int]:
        """
        Extracts the environment parameters from a discrete space.

        Args:
            space (gym.spaces.Discrete): The space to extract parameters from.

        Returns:
            Tuple[tuple, bool, int, int]: tuple containing (space_shape, space_continuous, space_min, space_max)
        """
        return (1,), False, space.start, space.n - 1

    @staticmethod
    def _get_params_from_box(space: gym.spaces.Box) -> Tuple[tuple, bool, List[float], List[float]]:
        """
        Extracts the environment parameters from a box space.

        Args:
            space (gym.spaces.Box): The space to extract parameters from.

        Returns:
            Tuple[tuple, bool, int, int]: tuple containing (space_shape, space_continuous, space_min, space_max)
        """
        return space.shape, True, space.low.tolist(), space.high.tolist()


class VmasWrapper(EnvironmentInterface):
    """
    Vectorized Multi-Agent Simulator (VMAS)

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

    def get_parameters(self) -> Union[EnvParams | MultiAgentEnvParams | MultiGroupEnvParams]:
        return self.env_params

    def reset(self, seed: int | None = None) -> TensorDict:
        obs = self.env.reset(seed=seed)

        # Stack the observation so it has shape (# env, # agents, obs shape)
        obs = torch.stack(obs, dim=1)
        state_td = TensorDict(
            {
                'observation': obs,
            },
            batch_size=torch.Size([self.env.batch_dim])
        )

        if self.render_mode == 'rgb_array':
            rgb = self.env.render(mode=self.render_mode)

            # Fix the negative stride in the numpy array
            img = rgb.copy()
            state_td['rgb_array'] = torch.from_numpy(img).unsqueeze(0)
        return state_td

    def step(self, action: TensorDict) -> TensorDict:
        # VMAS expects actions to have shape (# agents, # env, action shape)
        action_val = action['action'].permute(1, 0, 2)

        state, reward, done, info = self.env.step(action_val)
        state = torch.stack(state, dim=1)
        reward = torch.stack(reward, dim=1)
        action['next'] = {
            'observation': state,
            'reward': reward,
            'done': done.unsqueeze(-1),
        }

        if self.render_mode == 'rgb_array':
            rgb = self.env.render(mode=self.render_mode)

            # Fix the negative stride in the numpy array
            img = rgb.copy()
            action['next', 'rgb_array'] = torch.from_numpy(img).unsqueeze(0)

        return action

    def _make_env_params(self):
        # Get the agent names
        agent_names = [a.name for a in self.env.agents]

        # Extract group names by matching prefixes with the pattern 'agent_0', 'agent_1' and count the agents with the same prefix
        name_prefixes = Counter(item.rsplit('_', 1)[0] for item in agent_names)

        # Convert to a list of lists containing [[group_name, agent_count],[...]]
        group_list = [[key, count] for key, count in name_prefixes.items()]

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

        # Return a MultiAgentEnvParams object if there is only one agent group, otherwise return a MultiGroupEnvParams
        if len(group.keys()) == 1:
            return group[list(group.keys())[0]]
        else:
            return MultiGroupEnvParams(
                group=group,
            )
