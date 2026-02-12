"""
General policies that are applicable across various reinforcement learning algorithms.
"""
from .interface import Policy, InfoDict

__all__ = [
    "Policy",
    "InfoDict",
    "PolicyFactory",
]
