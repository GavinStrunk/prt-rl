from abc import ABC, abstractmethod
from typing import Optional, Tuple

class BaseEnvironment(ABC):
    """
    Defines the environment interface for JHU simulations.
    """
    @abstractmethod
    def get_number_of_states(self) -> int:
        """
        Returns the number of states in the environment.

        Returns:
            int: number of states
        """
        pass

    @abstractmethod
    def get_number_of_actions(self) -> int:
        """
        Returns the number of actions in the environment.

        Returns:
            int: number of actions
        """
        pass

    @abstractmethod
    def reset(self,
              seed: Optional[int] = None,
              randomize_start: Optional[bool] = False
              ) -> int:
        """
        Resets the environment to the initial state.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            randomize_start (bool, optional): Whether to randomize the starting state. Not all environments will support this. Defaults to False.

        Returns:
            int: current state value
        """
        pass

    @abstractmethod
    def execute_action(self,
                       action: int
                       ) -> Tuple[int, float, bool]:
        """
        Executes the agent's action and steps the simulation.

        Args:
            action (int): Action to execute

        Returns:
            Tuple[int, float, bool]: Next state, reward, done
        """
        pass

    def render(self):
        """
        Renders the environment
        """
        raise NotImplementedError()