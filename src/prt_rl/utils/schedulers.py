from abc import ABC, abstractmethod
import torch
from typing import Any

class ParameterScheduler(ABC):
    def __init__(self,
                 target_object: Any,
                 parameter_name: str
                 ):
        self.target_object = target_object
        self.parameter_name = parameter_name

    @abstractmethod
    def update(self,
               iteration_number: int
               ) -> None:
        raise NotImplementedError

class LinearScheduler(ParameterScheduler):
    def __init__(self,
                 target_object: Any,
                 parameter_name: str,
                 max_value: float,
                 min_value: float,
                 num_episodes: int
                 ) -> None:
        super(LinearScheduler, self).__init__(target_object=target_object, parameter_name=parameter_name)
        self.max_value = max_value
        self.min_value = min_value
        self.num_episodes = num_episodes
        self.rate = -(self.max_value - self.min_value) / self.num_episodes

    def update(self,
               iteration_number: int
               ) -> None:

        param_value = iteration_number * self.rate + self.max_value
        param_value = max(param_value, self.min_value)
        self.target_object.set_parameter(self.parameter_name, param_value)

class ExponentialScheduler(ParameterScheduler):
    def __init__(self,
                 target_object: Any,
                 parameter_name: str,
                 max_value: float,
                 min_value: float,
                 decay_rate: float,
                 ) -> None:
        super(ExponentialScheduler, self).__init__(target_object=target_object, parameter_name=parameter_name)
        self.max_value = max_value
        self.min_value = min_value
        self.decay_rate = decay_rate

    def update(self,
               iteration_number: int
               ) -> None:
        param_value = self.min_value + (self.max_value - self.min_value) * torch.exp(-self.decay_rate * iteration_number)
        self.target_object.set_parameter(self.parameter_name, param_value)