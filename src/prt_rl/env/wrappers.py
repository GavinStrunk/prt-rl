import gymnasium as gym
import numpy as np
from tensordict.tensordict import TensorDict
import torch
from typing import Optional, Tuple, List, Union
import vmas
from prt_sim.jhu.base import BaseEnvironment
from prt_sim.jhu.bandits import KArmBandits
from prt_rl.env.interface import EnvironmentInterface, EnvParams, MultiAgentEnvParams, MultiGroupEnvParams


class JhuWrapper(EnvironmentInterface):
    """
    Wraps the JHU environments in the Environment interface.

    The JHU environments are games and puzzles that were used in the JHU 705.741 RL course.
    """
    def __init__(self,
                 environment: BaseEnvironment,
                 render_mode: Optional[str] = None,
                 ) -> None:
        super().__init__(render_mode)
        self.env = environment

    def get_parameters(self) -> EnvParams:
        params = EnvParams(
            action_shape=(1,),
            action_continuous=False,
            action_min=0,
            action_max=self.env.get_number_of_actions()-1,
            observation_shape=(1,),
            observation_continuous=False,
            observation_min=0,
            observation_max=max(self.env.get_number_of_states()-1, 0),
        )
        return params

    def reset(self) -> TensorDict:
        state = self.env.reset()
        state_td = TensorDict(
            {
                'observation': torch.tensor([[state]], dtype=torch.int),
            },
            batch_size=torch.Size([1])
        )

        # Add info for Bandit environment
        if isinstance(self.env, KArmBandits):
            state_td['info'] = {
                'optimal_bandit': torch.tensor([[self.env.get_optimal_bandit()]], dtype=torch.int),
            }

        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            rgb = self.env.render()
            state_td['rgb_array'] = torch.tensor(rgb).unsqueeze(0)

        return state_td

    def step(self, action: TensorDict) -> TensorDict:
        action_val = action['action'][0].item()
        state, reward, done = self.env.execute_action(action_val)
        action['next'] = {
            'observation': torch.tensor([[state]], dtype=torch.int),
            'reward': torch.tensor([[reward]], dtype=torch.float),
            'done': torch.tensor([[done]], dtype=torch.bool),
        }

        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            rgb = self.env.render()
            action['next', 'rgb_array'] = torch.tensor(rgb).unsqueeze(0)

        return action

class GymnasiumWrapper(EnvironmentInterface):
    """
    Wraps the Gymnasium environments in the Environment interface.

    Args:
        gym_name: Name of the Gymnasium environment.
        render_mode: Sets the rendering mode. Defaults to None.

    """
    def __init__(self,
                 gym_name: str,
                 render_mode: Optional[str] = None,
                 **kwargs
                 ) -> None:
        super().__init__(render_mode)
        self.gym_name = gym_name
        self.env = gym.make(self.gym_name, render_mode=render_mode, **kwargs)
        self.env_params = self._make_env_params()

    def get_parameters(self) -> EnvParams:
        return self.env_params

    def reset(self) -> TensorDict:
        obs, info = self.env.reset()

        state_td = TensorDict(
            {
                'observation': self._process_observation(obs),
            },
            batch_size=torch.Size([1])
        )

        if self.render_mode == 'rgb_array':
            rgb = self.env.render()
            state_td['rgb_array'] = torch.tensor(rgb).unsqueeze(0)
        return state_td

    def step(self, action: TensorDict) -> TensorDict:
        action_val = action['action'][0]

        # Discrete actions send the raw integer value to the step function
        if not self.env_params.action_continuous:
            action_val = action_val.item()
        else:
            action_val = action_val.cpu().numpy()

        state, reward, terminated, trunc, info = self.env.step(action_val)
        done = terminated or trunc
        action['next'] = {
            'observation': self._process_observation(state),
            'reward': torch.tensor([[reward]], dtype=torch.float),
            'done': torch.tensor([[done]], dtype=torch.bool),
        }

        if self.render_mode == 'rgb_array':
            rgb = self.env.render()
            action['next', 'rgb_array'] = torch.tensor(rgb).unsqueeze(0)

        return action

    def _process_observation(self, observation) -> torch.Tensor:
        """
        Process the observation to handle different output types like int, list[int] to standardize them into a tensor.

        """
        if isinstance(observation, int):
            observation = np.array([observation])

        return torch.tensor(observation).unsqueeze(0)

    def _make_env_params(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            act_shape, act_cont, act_min, act_max = self._get_params_from_discrete(self.env.action_space)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            act_shape, act_cont, act_min, act_max = self._get_params_from_box(self.env.action_space)
        else:
            raise NotImplementedError(f"{self.env.action_space} action space is not supported")

        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            obs_shape, obs_cont, obs_min, obs_max = self._get_params_from_discrete(self.env.observation_space)
        elif isinstance(self.env.observation_space, gym.spaces.Box):
            obs_shape, obs_cont, obs_min, obs_max = self._get_params_from_box(self.env.observation_space)
        else:
            raise NotImplementedError(f"{self.env.observation_space} observation space is not supported")

        return EnvParams(
            action_shape=act_shape,
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

    def reset(self) -> TensorDict:
        obs = self.env.reset()

        state_td = TensorDict(
            {
                'observation': self._process_observation(obs),
            },
            batch_size=torch.Size([self.env.batch_dim])
        )

        if self.render_mode == 'rgb_array':
            rgb = self.env.render(mode=self.render_mode)
            state_td['rgb_array'] = torch.tensor(rgb).unsqueeze(0)
        return state_td


    def step(self, action: TensorDict) -> TensorDict:
        pass

    def _make_env_params(self):
        return MultiAgentEnvParams()