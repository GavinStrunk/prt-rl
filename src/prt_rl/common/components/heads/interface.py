from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Tuple


class DistributionHead(ABC, nn.Module):
    """
    Interface for distribution heads that output a distribution over actions.
    """
    @abstractmethod
    @torch.no_grad()
    def sample(self, latent: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample an action from the distribution given the latent representation.

        Args:
            latent (Tensor): The input to the head, typically the output of a backbone network.
            deterministic (bool): Whether to sample deterministically (e.g., take the mean) or stochastically.
        Returns:
            action (Tensor): The sampled action.
            log_prob (Tensor): The log probability of the sampled action.
            entropy (Tensor): The entropy of the distribution.
        """
        raise NotImplementedError("The sample method must be implemented by subclasses.")
    
    def log_prob(self, latent: Tensor, action: Tensor) -> Tensor:
        """
        Compute the log probability of a given action under the distribution defined by the latent representation.

        Args:
            latent (Tensor): The input to the head, typically the output of a backbone network.
            action (Tensor): The action for which to compute the log probability.
        Returns:
            log_prob (Tensor): The log probability of the given action.
        """
        raise NotImplementedError("The log_prob method must be implemented by subclasses.")
    
    def entropy(self, latent: Tensor) -> Tensor:
        """
        Compute the entropy of the distribution defined by the latent representation.

        Args:
            latent (Tensor): The input to the head, typically the output of a backbone network.
        Returns:
            entropy (Tensor): The entropy of the distribution.
        """
        raise NotImplementedError("The entropy method must be implemented by subclasses.")
    
