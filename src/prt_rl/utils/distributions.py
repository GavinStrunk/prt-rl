from abc import ABC, abstractmethod
import torch
import torch.distributions as dist

class Distribution(ABC):
    @staticmethod
    @abstractmethod
    def parameters_per_action() -> int:
        pass

class Categorical(Distribution, dist.Categorical):
    def __init__(self,
                 probs: torch.Tensor
                 ) -> None:
        # Probabilities are passed in with shape (# batch, # actions, # params)
        # Categorical only has 1 param and wants the list with shape (# batch, # action probs) so we squeeze the last dimension
        probs = probs.squeeze(-1)
        super().__init__(probs)

    @staticmethod
    def parameters_per_action() -> int:
        return 1

class Normal(Distribution, dist.Normal):
    def __init__(self,
                 probs: torch.Tensor
                 ) -> None:
        scale = torch.nn.functional.relu(probs[..., 1])
        super().__init__(loc=probs[..., 0], scale=scale)

    @staticmethod
    def parameters_per_action() -> int:
        return 2