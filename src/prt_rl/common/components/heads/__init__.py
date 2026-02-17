from .heads import ValueHead, QValueHead, DuelingHead, GaussianHead, BetaHead, CategoricalHead
from .interface import DistributionHead

__all__ = [
  "DistributionHead",
  "BetaHead",
  "CategoricalHead",
  "DuelingHead",
  "GaussianHead",
  "QValueHead",
  "ValueHead"
  ]