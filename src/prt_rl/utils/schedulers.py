from abc import ABC, abstractmethod
import numpy as np

class ParameterScheduler(ABC):
    """
    Abstract class for parameter scheduling.

    Args:
        parameter_name (str): Name of the parameter to schedule
    """
    def __init__(self,
                 parameter_name: str
                 ):
        self.parameter_name = parameter_name

    @abstractmethod
    def update(self,
               iteration_number: int
               ) -> float:
        """
        Returns the updated parameter value based on the current iteration number.

        Args:
            iteration_number (int): Current iteration number

        Returns:
            float: Updated parameter value
        """
        raise NotImplementedError

class LinearScheduler(ParameterScheduler):
    """
    Linear schedule updates a parameter from a maximum value to a minimum value over a given number of episodes.

    Args:
        parameter_name (str): Name of the parameter to schedule
        start_value (float): Maximum value for the parameter
        end_value (float): Minimum value for the parameter
        num_episodes (int): Number of episodes to compute schedule over

    """
    def __init__(self,
                 parameter_name: str,
                 start_value: float,
                 end_value: float,
                 num_episodes: int
                 ) -> None:
        super(LinearScheduler, self).__init__(parameter_name=parameter_name)
        assert num_episodes > 0, "Number of episodes must be greater than 0"

        self.start_value = start_value
        self.end_value = end_value
        self.num_episodes = num_episodes
        self.rate = -(self.start_value - self.end_value) / self.num_episodes

    def update(self,
               iteration_number: int
               ) -> float:
        """
        Returns the linearly scheduled parameter value based on the current iteration number.

        Args:
            iteration_number (int): Current iteration number

        Returns:
            float: Linearly scheduled parameter value
        """
        param_value = iteration_number * self.rate + self.start_value
        param_value = max(param_value, self.end_value) if self.rate < 0 else min(param_value, self.end_value)
        return param_value

class ExponentialScheduler(ParameterScheduler):
    """
    Exponential scheduler updates a parameter from a maximum value to a minimum value with a given exponential decay.

    Args:
        parameter_name (str): Name of the parameter to schedule
        start_value (float): Maximum value for the parameter
        end_value (float): Minimum value for the parameter
        decay_rate (float): Exponential decay rate for the parameter
    """
    def __init__(self,
                 parameter_name: str,
                 start_value: float,
                 end_value: float,
                 decay_rate: float,
                 ) -> None:
        super(ExponentialScheduler, self).__init__(parameter_name=parameter_name)
        self.start_value = start_value
        self.end_value = end_value
        self.decay_rate = decay_rate

    def update(self,
               iteration_number: int
               ) -> float:
        """
        Returns the updated parameter value based on the current iteration number.

        Args:
            iteration_number (int): Current iteration number

        Returns:
            float: Updated parameter value
        """
        param_value = self.end_value + (self.start_value - self.end_value) * np.exp(-self.decay_rate * iteration_number)
        return param_value