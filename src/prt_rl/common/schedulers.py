from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple, List

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
        end_value (Union[float, List[float]]): Minimum value for the parameter
        interval (Union[int , Tuple[int, int], List[Tuple[int, int]]]): Interval to schedule the parameter over. Can be a single integer, a tuple of integers, or a list of tuples. If a single integer is provided, the parameter will be scheduled over that many episodes. If a tuple is provided, the parameter will be scheduled over that range of episodes. If a list of tuples is provided, the parameter will be scheduled over each interval in the list.

    Raises:
        ValueError: If the interval is not greater than 0 or if the length of end_value and interval are not the same

    Example:
        .. python::
            from prt_rl.common.schedulers import LinearScheduler
            from prt_rl.common.epsilon_greedy import EpsilonGreedy
            eg = EpsilonGreedy()

            # Schedule epsilon from 0.2 to 0.1 over 10 episodes starting from episode 0
            s = LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.2, end_value=0.1, interval=10)

            # Schedule epsilon over an interval of (4, 10) from 0.2 to 0.1
            s = LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.2, end_value=0.1, interval=(4, 10))

            # Piecewise schedule epsilon over multiple intervals
            s = LinearScheduler(obj=eg, parameter_name='epsilon', start_value=1.0, end_value=[0.1, 0.01], interval=[(0, 10), (15, 20)])
    """
    def __init__(self,
                 obj: object,
                 parameter_name: str,
                 start_value: float,
                 end_value: Union[float, List[float]],
                 interval: Union[int , Tuple[int, int], List[Tuple[int, int]]],
                 ) -> None:
        super(LinearScheduler, self).__init__(obj=obj, parameter_name=parameter_name)
        if isinstance(end_value, float):
            end_value = [end_value]

        if isinstance(interval, int):
            if interval <= 0:
                raise ValueError("Interval must be greater than 0")
            interval = (0, interval)

        if isinstance(interval, tuple):
            interval = [interval]

        if len(end_value) != len(interval):
            raise ValueError(f"Length of end_value {len(end_value)} and interval {len(interval)} must be the same")
        
        self.check_intervals(interval)

        self.start_value = start_value
        self.end_value = end_value
        self.interval = interval
        self.current_value = start_value

        # Calculate the rates for each interval
        values = [self.start_value] + end_value
        value_steps = [values[i+1] - values[i] for i in range(len(values)-1)]
        interval_steps = [i[1] - i[0] for i in interval]
        self.rates = [x/y for x, y in zip(value_steps, interval_steps)]

    def update(self,
               current_step: int
               ) -> None:
        """
        Returns the linearly scheduled parameter value based on the current step number.

        Args:
            current_step (int): Current step number
        """
        # Check if the current step is within any of the intervals
        for i, (start, end) in enumerate(self.interval):
            if start <= current_step <= end:
                if i == 0:
                    start_val = self.start_value
                else:
                    start_val = self.end_value[i-1]
                    
                self.current_value = (current_step - start) * self.rates[i] + start_val
                self.current_value = max(self.current_value, self.end_value[i]) if self.rates[i] < 0 else min(self.current_value, self.end_value[i])
                break
        
        setattr(self.obj, self.parameter_name, self.current_value)

    def check_intervals(self, intervals: list[tuple[int, int]]) -> None:
        """
        Check if the intervals are overlapping.

        Args:
            intervals (list[tuple[int, int]]): List of intervals to check
        
        Raises:
            ValueError: If any of the intervals overlap
        """
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                if self._is_interval_overlapping(intervals[i], intervals[j]):
                    raise ValueError(f"Intervals {intervals[i]} and {intervals[j]} overlap.")

    def _is_interval_overlapping(self, interval1: tuple[int, int], interval2: tuple[int, int]) -> bool:
        """
        Check if two intervals overlap.
        
        Args:
            interval1 (tuple[int, int]): First interval
            interval2 (tuple[int, int]): Second interval
        Returns:
            bool: True if the intervals overlap, False otherwise
        """
        a_start, a_end = interval1
        b_start, b_end = interval2
        return max(a_start, b_start) < min(a_end, b_end)

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