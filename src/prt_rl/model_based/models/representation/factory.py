from abc import ABC, abstractmethod
from pathlib import Path
import torch
from typing import TypeVar, Generic

from prt_rl.env.interface import EnvParams
from prt_rl.model_based.models.representation.interface import RepresentationInterface

TSpec = TypeVar("TSpec")
TModel = TypeVar("TModel", bound=RepresentationInterface)

class RepresentationModelFactory(ABC, Generic[TSpec, TModel]):
    """
    Abstract factory for creating, saving, and loading representation models. Each algorithm should implement its own factory.
    The factory should take an algorithm specification and environment parameters to create a representation model instance that is valid for the specific algorithm.
    The factory must also specify how to save and load the model so the agent can checkpoint and reconstruct the generic model.

    Factory owns:
      - make(env, spec) -> model
      - save(env, spec, model, path)
      - load(path) -> model
    """
    @abstractmethod
    def make(self, env_params: EnvParams, spec: TSpec) -> TModel:
        """
        Creates a representation model instance based on the provided environment parameters and algorithm specification.
        
        Args:
            env (EnvParams): Environment parameters.
            spec (TSpec): Algorithm specification.
        Returns:
            TModel: An instance of the representation model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def save(self, env_params: EnvParams, spec: TSpec, model: TModel, path: str | Path) -> None:
        """
        Saves the representation model to the specified path along with environment parameters and algorithm specification.
        
        Args:
            env (EnvParams): Environment parameters.
            spec (TSpec): Algorithm specification.
            model (TModel): The representation model instance to be saved.
            path (str | Path): The file path where the model should be saved.
        """
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> TModel:
        """
        Loads the representation model from the specified path.
        
        Args:
            path (str | Path): The file path from where the model should be loaded.
            map_location (str | torch.device): The device to map the loaded model to.
        Returns:
            TModel: An instance of the loaded representation model.
        """
        raise NotImplementedError