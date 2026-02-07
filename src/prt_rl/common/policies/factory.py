"""
Abstract factory for creating, saving, and loading policies.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Tuple, TypeVar, Union
import torch

from prt_rl.env.interface import EnvParams
from prt_rl.common.policies.base import PolicyModule

TSpec = TypeVar("TSpec")
TPolicy = TypeVar("TPolicy", bound=PolicyModule)

class PolicyFactory(ABC, Generic[TSpec, TPolicy]):
    """
    Abstract factory for creating, saving, and loading policies. Each algorithm should implement its own factory.
    The factory should take an algorithm specification and environment parameters to create a policy instance that is valid for the specific algorithm.
    The factory must also specify how to save and load the policy so the agent can checkpoint and reconstruct the generic policy.
    
    Factory owns:
      - make(env, spec) -> policy
      - save(env, spec, policy, path)
      - load(path) -> (env, spec, policy)
    """

    @abstractmethod
    def make(self, env_params: EnvParams, spec: TSpec) -> TPolicy:
        """
        Creates a policy instance based on the provided environment parameters and algorithm specification.
        
        Args:
            env (EnvParams): Environment parameters.
            spec (TSpec): Algorithm specification.
        Returns:
            TPolicy: An instance of the policy.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, env_params: EnvParams, spec: TSpec, policy: TPolicy, path: Union[str, Path]) -> None:
        """
        Saves the policy to the specified path along with environment parameters and algorithm specification.
        
        Args:
            env (EnvParams): Environment parameters.
            spec (TSpec): Algorithm specification.
            policy (TPolicy): The policy instance to be saved.
            path (Union[str, Path]): The file path where the policy should be saved.
        """
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> Tuple[EnvParams, TSpec, TPolicy]:
        """
        Loads the policy from the specified path along with environment parameters and algorithm specification.
        
        Args:
            path (Union[str, Path]): The file path from where the policy should be loaded.
            map_location (Union[str, torch.device], optional): Device mapping for loading the policy. Defaults to "cpu".
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict. Defaults to True.
        
        Returns:
            Tuple[EnvParams, TSpec, TPolicy]: A tuple containing environment parameters, algorithm specification, and the loaded policy instance.
        """
        raise NotImplementedError 