from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from prt_rl.utils.qtable import QTable
from prt_rl.utils.decision_functions import DecisionFunction


class Policy(ABC):
    def __init__(self,
                 decision_function: DecisionFunction,):
        self.decision_function = decision_function

    @abstractmethod
    def get_action(self,
                   state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        if hasattr(self.decision_function, name):
            self.decision_function.set_parameter(name, value)
        else:
            raise ValueError(f"Parameter '{name}' not found in Policy.")

    @staticmethod
    def load_from_file(filename: str) -> 'Policy':
        raise NotImplementedError

    def save(self, filename: str):
        raise NotImplementedError

class QTablePolicy:
    def __init__(self,
                 q_table: QTable,
                 decision_function: DecisionFunction,
                 ):
        self.q_table = q_table
        self.decision_function = decision_function

    def get_action(self,
                   state: torch.Tensor) -> torch.Tensor:
        q_values = self.q_table.get_action_values(state)
        return self.decision_function.select_action(q_values)

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        if hasattr(self.q_table, name):
            setattr(self.q_table, name, value)
        elif hasattr(self.decision_function, name):
            self.decision_function.set_parameter(name, value)
        else:
            raise ValueError(f"Parameter '{name}' not found in QTablePolicy.")

class QNetworkPolicy:
    def __init__(self,
                 q_network: torch.nn.Module,
                 decision_function: DecisionFunction,
                 ) -> None:
        self.q_network = q_network
        self.decision_function = decision_function

    def get_action(self,
                   state: torch.Tensor) -> torch.Tensor:
        q_values = self.q_network.get_action(state)
        return self.decision_function.select_action(q_values)

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        if hasattr(self.q_network, name):
            setattr(self.q_network, name, value)
        elif hasattr(self.decision_function, name):
            self.decision_function.set_parameter(name, value)
        else:
            raise ValueError(f"Parameter '{name}' not found in QNetworkPolicy.")

def greedy(actions: list[int]) -> int:
    """
    Greedy policy chooses the action with the highest value.

    .. math::
        A_t \equiv argmax Q_t(a)

    Notes:
        If there are multiple actions with the same maximum value, they are sampled randomly to choose the action.

    Args:
        actions (list[int]): list of action values

    Returns:
        int: Selected action
    """
    # Select action with largest value, choose randomly if there is more than 1
    a_star = np.where(actions == np.max(actions))[0]
    if len(a_star) > 1:
        a_star = np.random.choice(a_star)
    else:
        a_star = a_star[0]

    return a_star

def epsilon_greedy(actions: list[int], epsilon: float) -> int:
    """
    Epsilon-greedy policy chooses the action with the highest value and samples all actions randomly with probability epsilon.

    If :math:`b > \epsilon` , use Greedy; otherwise choose randomly from amount all actions

    Args:
        actions (list[int]): list of action values
        epsilon (float): probability of choosing a random exploratory action

    Returns:
        int: Selected action
    """
    # Select action with largest value, choose randomly if there is more than 1
    a_star = np.where(actions == np.max(actions))[0]
    if len(a_star) > 1:
        a_star = np.random.choice(a_star)
    else:
        a_star = a_star[0]

    if np.random.random() > epsilon:
        action = a_star
    else:
        # Choose random from actions
        action = np.random.choice(len(actions))

    return action

def softmax(actions: list[int], tau: float) -> int:
    """
    Soft-Max policy models a Boltzmann (or Gibbs) distribution to select an action probabilistically with the highest value.

    $$Pr(A_t = a) \equiv \pi_t(a) \equiv \frac{a}{b}$$

    Args:
        actions (list[int]): list of action values
        tau (float): probability of choosing a random exploratory action

    Returns:
        int: Selected action
    """
    # Compute exponential values
    exp_values = np.exp(np.array(actions) / tau)

    # Normalize to get probabilities
    action_probs = exp_values / np.sum(exp_values)

    # Sample from the probabilities to get the action
    action = np.random.choice(len(actions), p=action_probs)

    return action

def upper_confidence_bound(actions: list[int], action_selections: list[int], c: float, t: int) -> int:
    """
    Upper Confidence Bound selects among the non-greedy actions based on their potential for being optimal.

    .. math::
        A_t \equiv argmax [Q_t(a) + c\sqrt{\frac{ln t}{N_t(a)}}

    Args:
        actions (list[int]): list of action values
        action_selections (int): Number of times the actions have been selected prior to time t
        c (float): Constant controlling degree of exploration
        t (int): time step

    Returns:
        int: Selected action
    """
    assert c > 0, "Constant controlling degree of exploration must be > 0"
    assert len(actions) == len(action_selections), "Action values and action selections must be same length"

    a_star = actions + c*np.sqrt((np.log(t) / action_selections))
    a_star = np.where(a_star == np.max(a_star))[0]
    if len(a_star) > 1:
        a_star = np.random.choice(a_star)
    else:
        a_star = a_star[0]

    return a_star