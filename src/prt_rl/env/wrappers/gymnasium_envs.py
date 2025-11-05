"""
Wrapper for Gymnasium environments.
"""
import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Tuple, List, Union, Dict, Any
from prt_rl.env.interface import EnvironmentInterface, EnvParams

class GymnasiumWrapper(EnvironmentInterface):
    """
    Wraps the Gymnasium environments in the Environment interface.

    Args:
        gym_name: Name of the Gymnasium environment.
        num_envs: Number of parallel environments to create.
        render_mode: Sets the rendering mode. Defaults to None.

    Examples:
        .. code-block:: python

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
        
        # If there is only one environment, reset it directly
        if self.num_envs == 1:
            state, info = self.reset(seed=seed)
        else:
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
            action = action.detach().cpu().numpy()

            # If there is only one environment remove the first dimension
            if action.shape[0] == 1:
                action = action[0]

        next_state, reward, terminated, trunc, info = self.env.step(action)
        done = np.logical_or(terminated, trunc)

        # Reshape the reward and done to be (# envs, 1)
        if self.num_envs == 1:
            reward = torch.tensor([[reward]], dtype=torch.float, device=self.device)
            done = torch.tensor([[bool(done)]], dtype=torch.bool, device=self.device)
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

        # If observation is float64 convert it to float32
        if observation.dtype == torch.float64:
            observation = observation.float()
            
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
            action_len, act_cont, act_min, act_max = self._get_params_from_discrete(action_space, is_action=True)
        elif isinstance(action_space, gym.spaces.Box):
            action_len, act_cont, act_min, act_max = self._get_params_from_box(action_space, is_action=True)
        elif isinstance(action_space, gym.spaces.Dict):
            action_len, act_cont, act_min, act_max = self._get_params_from_dict(action_space, is_action=True)
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
    def _get_params_from_discrete(space: gym.spaces.Discrete, is_action: bool = False) -> Tuple[tuple | int, bool, int, int]:
        """
        Extracts the environment parameters from a discrete space.

        Args:
            space (gym.spaces.Discrete): The space to extract parameters from.

        Returns:
            Tuple[tuple, bool, int, int]: tuple containing (space_shape, space_continuous, space_min, space_max)
        """
        # If this is a discrete action space return an integer action length
        if is_action:
            space_shape = 1
        else:
            space_shape = (1,)
        return space_shape, False, space.start, space.n - 1

    @staticmethod
    def _get_params_from_box(space: gym.spaces.Box, is_action: bool = False) -> Tuple[tuple, bool, List[float], List[float]]:
        """
        Extracts the environment parameters from a box space.

        Args:
            space (gym.spaces.Box): The space to extract parameters from.

        Returns:
            Tuple[tuple, bool, int, int]: tuple containing (space_shape, space_continuous, space_min, space_max)
        """
        space_shape = space.shape

        # Retun an integer action length for box action spaces
        if is_action and len(space_shape) == 1:
            space_shape = space_shape[0]

        return space_shape, True, space.low.tolist(), space.high.tolist()
    
    @staticmethod
    def _get_params_from_dict(space: gym.spaces.Dict, is_action: bool = False) -> Tuple[tuple, List[bool], List[float], List[float]]:
        """
        Extracts the environment parameters from a dict space by concatenating all subspaces.

        Args:
            space (gym.spaces.Dict): The space to extract parameters from.
        Returns:
            Tuple[tuple, List[bool], List[float], List[float]]: tuple containing (space_shape, space_continuous, space_min, space_max)
        """
        if is_action:
            action_lens = []
            action_conts = []
            action_mins = []
            action_maxs = []
            for k in space.spaces:
                subspace = space.spaces[k]
                if isinstance(subspace, gym.spaces.Discrete):
                    alen, acont, amin, amax = GymnasiumWrapper._get_params_from_discrete(subspace, is_action=True)
                elif isinstance(subspace, gym.spaces.Box):
                    alen, acont, amin, amax = GymnasiumWrapper._get_params_from_box(subspace, is_action=True)
                    if isinstance(acont, bool):
                        acont = [acont] * alen
                else:
                    raise NotImplementedError(f"{subspace} action space is not supported in Dict")
                action_lens.append(alen)
                action_conts.extend(acont if isinstance(acont, list) else [acont])
                action_mins.extend(amin if isinstance(amin, list) else [amin])
                action_maxs.extend(amax if isinstance(amax, list) else [amax])

            total_action_len = sum(action_lens)

            return total_action_len, action_conts, action_mins, action_maxs
        else:
            raise NotImplementedError("Dict observation spaces are not supported")