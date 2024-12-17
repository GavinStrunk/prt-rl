from abc import abstractmethod, ABC
import torch
from typing import Any

class DecisionFunction(ABC):
    """
    A decision function takes in the state-action values from a Q function and returns a selected action.
    """
    @abstractmethod
    def select_action(self, action_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise ValueError(f"Parameter '{name}' not found.")

class Greedy(DecisionFunction):
    """
    Greedy policy chooses the action with the highest value.

    .. math::
        A_t \equiv argmax Q_t(a)

    Notes:
        If there are multiple actions with the same maximum value, they are sampled randomly to choose the action.

    Args:
        action_values (torch.Tensor): 1D tensor of state-action values.

    Returns:
        torch.Tensor: Selected action index.
    """
    def select_action(self, action_values: torch.Tensor) -> torch.Tensor:
        if action_values.ndim != 1:
            raise ValueError(
                "Expected a 1D tensor for actions, but got a tensor with shape: {}".format(action_values.shape))

        # Find indices of the maximum value(s)
        max_value = torch.max(action_values)
        max_indices = torch.nonzero(action_values == max_value, as_tuple=False).squeeze(-1)

        # Randomly select one if there are multiple maximum indices
        if len(max_indices) > 1:
            random_index = torch.randint(len(max_indices), (1,), device=action_values.device)
            selected_action = max_indices[random_index]
        else:
            selected_action = max_indices[0]

        return selected_action

class EpsilonGreedy(DecisionFunction):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_action(self, action_values: torch.Tensor) -> torch.Tensor:
        """
        Epsilon-greedy policy chooses the action with the highest value and samples all actions randomly with probability epsilon.

        If :math:`b > \epsilon`, use Greedy; otherwise choose randomly from among all actions.

        Args:
            actions (torch.Tensor): 1D tensor of action values.
            epsilon (float): Probability of choosing a random exploratory action.

        Returns:
            torch.Tensor: Selected action index.
        """
        if action_values.ndim != 1:
            raise ValueError("Expected a 1D tensor for actions, but got a tensor with shape: {}".format(actions.shape))

        # Determine device
        device = action_values.device

        # Greedy part: Find indices of the maximum value(s)
        max_value = torch.max(action_values)
        max_indices = torch.nonzero(action_values == max_value, as_tuple=False).squeeze(-1)

        # Randomly select one if there are multiple maximum indices
        if len(max_indices) > 1:
            random_index = torch.randint(len(max_indices), (1,), device=device)
            greedy_action = max_indices[random_index]
        else:
            greedy_action = max_indices[0]

        # Epsilon-greedy logic
        if torch.rand(1, device=device).item() > self.epsilon:
            # Exploit: Choose greedy action
            action = greedy_action
        else:
            # Explore: Choose a random action
            random_action = torch.randint(len(action_values), (1,), device=device)
            action = random_action[0]

        return action

class Softmax(DecisionFunction):
    def __init__(self, tau: float):
        self.tau = tau

    def select_action(self, action_values: torch.Tensor) -> torch.Tensor:
        """
        Softmax policy models a Boltzmann (or Gibbs) distribution to select an action probabilistically with the highest value.

        Args:
            actions (torch.Tensor): 1D tensor of action values.
            tau (float): Temperature parameter controlling exploration.

        Returns:
            torch.Tensor: Selected action index.
        """
        if action_values.ndim != 1:
            raise ValueError("Expected a 1D tensor for actions, but got a tensor with shape: {}".format(action_values.shape))

        # Compute exponential values scaled by tau
        exp_values = torch.exp(action_values / self.tau)

        # Normalize to get probabilities
        action_probs = exp_values / torch.sum(exp_values)

        # Sample from the probabilities to get the action
        action = torch.multinomial(action_probs, 1).squeeze(0)

        return action


class StochasticSelection:
    def select_action(self, action_pmf: torch.Tensor) -> torch.Tensor:
        """
        Perform a stochastic selection of an action based on a given PMF.

        Args:
            action_pmf (torch.Tensor): 1D tensor containing probabilities for each action.
                                       Must sum to 1 and have non-negative values.

        Returns:
            torch.Tensor: The index of the selected action.
        """
        if action_pmf.ndim != 1:
            raise ValueError(f"Expected a 1D tensor for action PMF, but got a tensor with shape: {action_pmf.shape}")

        if not torch.isclose(action_pmf.sum(), torch.tensor(1.0, device=action_pmf.device)):
            raise ValueError("The probabilities in the PMF must sum to 1.")

        if (action_pmf < 0).any():
            raise ValueError("The PMF cannot contain negative probabilities.")

        # Use torch.multinomial for stochastic sampling
        selected_action = torch.multinomial(action_pmf, 1).squeeze(0)

        return selected_action


class UpperConfidenceBound(DecisionFunction):
    def __init__(self,
                 c: float,
                 t: float
                 ):
        self.c = c
        self.t = t

    def select_action(self, action_values: torch.Tensor) -> torch.Tensor:
        """
        Upper Confidence Bound selects among the non-greedy actions based on their potential for being optimal.

        .. math::
            A_t \equiv argmax [Q_t(a) + c\sqrt{\frac{ln t}{N_t(a)}}

        Args:
            actions (torch.Tensor): 1D tensor of action values.
            action_selections (torch.Tensor): 1D tensor of the number of times each action has been selected.
            c (float): Constant controlling degree of exploration.
            t (int): Current time step.

        Returns:
            torch.Tensor: Selected action index.
        """
        if action_values.ndim != 1 or action_selections.ndim != 1:
            raise ValueError("Expected 1D tensors for actions and action_selections.")
        if action_values.shape != action_selections.shape:
            raise ValueError("Actions and action_selections must have the same shape.")
        if c <= 0:
            raise ValueError("The constant 'c' must be greater than 0.")

        # Compute UCB values
        log_term = torch.log(torch.tensor(self.t, dtype=torch.float32, device=action_values.device))
        exploration_bonus = self.c * torch.sqrt(log_term / action_selections)
        ucb_values = action_values + exploration_bonus

        # Find indices of the maximum value(s)
        max_value = torch.max(ucb_values)
        max_indices = torch.nonzero(ucb_values == max_value, as_tuple=False).squeeze(-1)

        # Randomly select one if there are multiple maximum indices
        if len(max_indices) > 1:
            random_index = torch.randint(len(max_indices), (1,), device=action_values.device)
            selected_action = max_indices[random_index]
        else:
            selected_action = max_indices[0]

        return selected_action
