from abc import ABC, abstractmethod
import numpy as np

class ParameterScheduler(ABC):
    """
    Abstract class for parameter scheduling.

    Args:
        obj (object): Object to which the parameter belongs
        parameter_name (str): Name of the parameter to schedule
    """
    def __init__(self,
                 obj: object,
                 parameter_name: str
                 ):
        self.obj = obj
        self.parameter_name = parameter_name

    @abstractmethod
    def update(self,
               current_step: int
               ) -> None:
        """
        Returns the updated parameter value based on the current step number.

        Args:
            current_step (int): Current step number
        """
        raise NotImplementedError

class LinearScheduler(ParameterScheduler):
    """
    Linear schedule updates a parameter from a maximum value to a minimum value over a given number of episodes.

    Args:
        obj (object): Object to which the parameter belongs
        parameter_name (str): Name of the parameter to schedule
        start_value (float): Maximum value for the parameter
        end_value (float): Minimum value for the parameter
        num_steps (int): Number of episodes to schedule the parameter
    """
    def __init__(self,
                 obj: object,
                 parameter_name: str,
                 start_value: float,
                 end_value: float,
                 num_steps: int
                 ) -> None:
        super(LinearScheduler, self).__init__(obj=obj, parameter_name=parameter_name)
        assert num_steps > 0, "Number of episodes must be greater than 0"

        self.start_value = start_value
        self.end_value = end_value
        self.num_episodes = num_steps
        self.rate = -(self.start_value - self.end_value) / self.num_episodes

    def update(self,
               current_step: int
               ) -> None:
        """
        Returns the linearly scheduled parameter value based on the current step number.

        Args:
            current_step (int): Current step number
        """
        param_value = current_step * self.rate + self.start_value
        param_value = max(param_value, self.end_value) if self.rate < 0 else min(param_value, self.end_value)
        setattr(self.obj, self.parameter_name, param_value)

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
                 obj: object,
                 parameter_name: str,
                 start_value: float,
                 end_value: float,
                 decay_rate: float,
                 ) -> None:
        super(ExponentialScheduler, self).__init__(obj=obj, parameter_name=parameter_name)
        self.start_value = start_value
        self.end_value = end_value
        self.decay_rate = decay_rate

    def update(self,
               current_step: int
               ) -> None:
        """
        Returns the updated parameter value based on the current step number.

        Args:
            current_step (int): Current step number
        """
        param_value = self.end_value + (self.start_value - self.end_value) * np.exp(-self.decay_rate * current_step)
        setattr(self.obj, self.parameter_name, param_value)