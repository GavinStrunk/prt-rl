from abc import ABC, abstractmethod
import torch.distributions as dist

class Distribution(ABC):
    @staticmethod
    @abstractmethod
    def parameters_per_action() -> int:
        pass

class Categorical(Distribution, dist.Categorical):
    @staticmethod
    def parameters_per_action() -> int:
        return 1

class Normal(Distribution, dist.Normal):
    @staticmethod
    def parameters_per_action() -> int:
        return 2