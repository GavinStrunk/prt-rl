from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch
from typing import Union, Optional, List, Dict, Tuple, Any


@dataclass
class EnvParams:
    """
    Environment parameters contains information about the action and observation spaces to configure RL algorithms.

    Parameters:
        action_len (int): Number of actions in action space
        action_continuous (bool): True if the actions are continuous or False if they are discrete
        action_min: Minimum action value. If the actions are discrete this is the minimum integer value, if the actions are continuous it matches the action shape with the minimum value for each action
        action_max: Maximum action values. If the actions are discrete this is the maximum integer value, if the actions are continuous it matches the action shape with the maximum value for each action
        observation_shape (tuple): shape of the observation space
        observation_continuous (bool): True if the observations are continuous or False if they are discrete
        observation_min: Minimum observation value. If the observations are discrete this is the minimum integer value, if the observations are continuous it matches the observation shape with the minimum value for each observation
        observation_max: Maximum observation value. If the observations are discrete this is the maximum integer value, if the observations are continuous it matches the observation shape with the maximum value for each observation
    """
    action_len: int
    action_continuous: bool
    action_min: Union[int, float, List[float]]
    action_max: Union[int, float, List[float]]
    observation_shape: tuple
    observation_continuous: bool
    observation_min: Union[int, float, List[float]]
    observation_max: Union[int, float, List[float]]

    def get_action_min_tensor(self) -> torch.Tensor:
        """
        Converts `action_min` to a tensor of shape (action_len, 1).
        - If `action_min` is a float, it is broadcast across all actions.
        - If it is a list, its length must match `action_len`.
        """
        if isinstance(self.action_min, float):
            return torch.full((self.action_len, 1), self.action_min)
        elif isinstance(self.action_min, list):
            if len(self.action_min) != self.action_len:
                raise ValueError(f"Expected action_min list to have length {self.action_len}, got {len(self.action_min)}")
            return torch.tensor(self.action_min, dtype=torch.float32).view(self.action_len, 1)
        else:
            raise TypeError("action_min must be a float or a list of floats.") 

    def get_action_max_tensor(self) -> torch.Tensor:
        """
        Converts `action_max` to a tensor of shape (action_len, 1).
        - If `action_max` is a float, it is broadcast across all actions.
        - If it is a list, its length must match `action_len`.
        """
        if isinstance(self.action_max, float):
            return torch.full((self.action_len, 1), self.action_max)
        elif isinstance(self.action_max, list):
            if len(self.action_max) != self.action_len:
                raise ValueError(f"Expected action_max list to have length {self.action_len}, got {len(self.action_max)}")
            return torch.tensor(self.action_max, dtype=torch.float32).view(self.action_len, 1)
        else:
            raise TypeError("action_max must be a float or a list of floats.")   

@dataclass
class MultiAgentEnvParams:
    """
    Multi-Agent environment parameters contains information about the action and observation spaces to configure multi-agent RL algorithms.

    Notes:
        This is still a work in progress.

    group = {
    name: (num_agents, EnvParams)
    }
    """
    num_agents: int
    agent: EnvParams


@dataclass
class MultiGroupEnvParams:
    """
    Multi-group environment parameters extends the Multi-agent parameters to group agents of the same type together. This allows heterogenous multi-agent teams to be trained together.

    """
    group: Dict[str, MultiAgentEnvParams]

class EnvironmentInterface(ABC):
    """
    The environment interface wraps other simulation environments to provide a consistent interface for the RL library.

    The interface for agents is based around tensors and a Gymnasium like API. The main extension to the gym API is the addition of the environment parameters and the ability to put the rgb_array in the info dictionary for rendering. 

    Single Agent Interface
    For a single agent step function returns the following structure:
    next_state, reward, done, info = env.step(action)

    The shape of each tensor is (N, M) where N is the number of environments and M is the size of the value. For example, if an agent has two output actions and we are training with four environments then the "action" key will have shape (4,2).

    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    def __init__(self,
                 render_mode: Optional[str] = None,
                 num_envs: int = 1,
                 ) -> None:
        self.render_mode = render_mode
        self.num_envs = num_envs

        if self.render_mode is not None:
            assert self.render_mode in EnvironmentInterface.metadata["render_modes"], f"Valid render_modes are: {EnvironmentInterface.metadata['render_modes']}"

    def get_num_envs(self) -> int:
        """
        Returns the number of environments in the interface.

        Returns:
            int: Number of environments
        """
        return self.num_envs

    @abstractmethod
    def get_parameters(self) -> Union[EnvParams]:
        """
        Returns the EnvParams object which contains information about the sizes of observations and actions needed for setting up RL agents.

        Returns:
            EnvParams: environment parameters object
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Resets the environment to the initial state and returns the initial observation.

        Args:
            seed (int | None): Sets the random seed.

        Returns:
            Tuple: Tuple of tensors containing the initial observation and info dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Steps the simulation using the action tensor and returns the new trajectory.

        Args:
            action (torch.Tensor): Tensor with "action" key that is a tensor with shape (# env, # actions)

        Returns:
            Tuple: Tuple of tensors containing the next state, reward, done, and info dictionary
        """
        raise NotImplementedError()
    
    def close(self) -> None:
        """
        Closes the environment and cleans up any resources.
        """
        pass


class NumpyEnvironmentInterface(ABC):
    """
    The NumpyEnvironmentInterface is a wrapper for environments that use numpy arrays for observations and actions. This is useful for environments that are not based on PyTorch or TensorFlow.

    The interface for agents is based around tensors and a Gymnasium like API. The main extension to the gym API is the addition of the environment parameters and the ability to put the rgb_array in the info dictionary for rendering. 

    Single Agent Interface
    For a single agent step function returns the following structure:
    next_state, reward, done, info = env.step(action)

    The shape of each tensor is (N, M) where N is the number of environments and M is the size of the value. For example, if an agent has two output actions and we are training with four environments then the "action" key will have shape (4,2).

    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    def __init__(self,
                 render_mode: Optional[str] = None,
                 ) -> None:
        self.render_mode = render_mode

        if self.render_mode is not None:
            assert self.render_mode in EnvironmentInterface.metadata["render_modes"], f"Valid render_modes are: {EnvironmentInterface.metadata['render_modes']}"

    @abstractmethod
    def get_parameters(self) -> Union[EnvParams]:
        """
        Returns the EnvParams object which contains information about the sizes of observations and actions needed for setting up RL agents.

        Returns:
            EnvParams: environment parameters object
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self, seed = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to the initial state and returns the initial observation.
        Args:
            seed (int | None): Sets the random seed.
        Returns:
            Tuple: Tuple of numpy array containing the initial observation and info dictionary
        """
        raise NotImplementedError()
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Steps the simulation using the action tensor and returns the new trajectory.

        Args:
            action (np.ndarray): Numpy array with "action" key that is a tensor with shape (# env, # actions)

        Returns:
            Tuple: Tuple of tensors containing the next state, reward, done, and info dictionary
        """
        raise NotImplementedError()