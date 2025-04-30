from abc import abstractmethod, ABC
import torch
from typing import Any


def stochastic_selection(action_pmf: torch.Tensor) -> torch.Tensor:
    """
    Perform a stochastic selection of an action based on a given PMF.

    Samples \pi(a \mid s) \rightarrow a

    Args:
        action_pmf (torch.Tensor): 1D tensor containing probabilities for each action.
                                   Must sum to 1 and have non-negative values.

    Returns:
        torch.Tensor: The index of the selected action.
    """
    if action_pmf.ndim != 2:
        raise ValueError(
            f"Expected a 2D tensor (# env, action_pmf) for action PMF, but got a tensor with shape: {action_pmf.shape}")

    if not torch.isclose(action_pmf.sum(dim=1), torch.ones(action_pmf.shape[0], device=action_pmf.device)).all():
        raise ValueError("The probabilities in the PMF must sum to 1.")

    if (action_pmf < 0).any():
        raise ValueError("The PMF cannot contain negative probabilities.")

    # Use torch.multinomial for stochastic sampling
    selected_action = torch.multinomial(action_pmf, 1)

    return selected_action


class DecisionFunction(ABC):
    """
    A decision function takes in the state-action values from a Q function and returns a selected action.

    Input:
    Tensor of action values with shape (# env, # action values)

    Output:
    Tensor of selected actions with shape (# env, 1)
    """

    @abstractmethod
    def select_action(self, action_values: torch.Tensor) -> torch.Tensor:
        """
        Selects an action from a vector of q values.

        Args:
            action_values (torch.Tensor): tensor of q values with shape (# environments, # actions)

        Returns:
            torch.Tensor: tensor of selected actions with shape (# environments, 1)
        """
        raise NotImplementedError

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        """
        Sets a named parameter in the decision function. This is used to set or update parameters values for example epsilon in epsilon-greedy.

        Args:
            name (str): name of the parameter
            value (Any): value to set

        """
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise ValueError(f"Parameter '{name}' not found.")

    @classmethod
    def from_dict(cls, data: dict) -> 'DecisionFunction':
        """
        Reconstruct the decision function from a dictionary.
        Child classes should override this if they have custom parameters.

        Args:
            data (dict): dictionary containing parameter values

        Returns:
            DecisionFunction: Decision function object
        """
        if data["type"] != cls.__name__:
            raise ValueError(f"Cannot load {data['type']} as {cls.__name__}")
        return cls()

    def to_dict(self) -> dict:
        """
        Serialize the decision function to a dictionary.
        Child classes should override this if they have custom parameters.

        Returns:
            dict: dictionary containing class type and parameter values
        """
        return {"type": self.__class__.__name__}


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

    def select_action(self,
                      action_values: torch.Tensor
                      ) -> torch.Tensor:
        if action_values.ndim != 2:
            raise ValueError(
                "Expected a tensor with shape (# env, # actions) for actions, but got a tensor with shape: {}".format(
                    action_values.shape))

        # Find indices of the maximum value(s)
        max_value, _ = torch.max(action_values, dim=1)

        # Find all indices where the value equals the max value
        max_indices_list = [
            (action_values[n] == max_value[n]).nonzero(as_tuple=True)[0].tolist()
            for n in range(action_values.size(0))
        ]

        # Randomly choose one index from the list of max indices for each dimension along N
        chosen_indices = torch.tensor([
            indices[torch.randint(len(indices), (1,)).item()] if len(indices) > 1 else indices[0]
            for indices in max_indices_list
        ]).unsqueeze(-1)

        return chosen_indices


class EpsilonGreedy(Greedy):
    """
    Epsilon-greedy is a soft policy version of greedy action selection, where a random action is chosen with probability epsilon and the maximum value action otherwise.

    Parameters:
        epsilon (float): probability of selecting a random action

    Args:
        epsilon (float): probability of selecting a random action
    """

    def __init__(self,
                 epsilon: float
                 ) -> None:
        self.epsilon = epsilon

    def select_action(self,
                      action_values: torch.Tensor
                      ) -> torch.Tensor:
        """
        Epsilon-greedy policy chooses the action with the highest value and samples all actions randomly with probability epsilon.

        If :math:`b > \epsilon`, use Greedy; otherwise choose randomly from among all actions.

        Args:
            action_values (torch.Tensor): Tensor of action values.

        Returns:
            torch.Tensor: Selected action index.
        """
        # Greedy action selection
        greedy_actions = Greedy.select_action(self, action_values)

        # Epsilon-greedy logic
        # Generate random values and check if they are larger than epsilon
        random_actions = torch.rand(action_values.size(0), device=action_values.device) <= self.epsilon
        actions = torch.zeros((action_values.shape[0], 1), device=action_values.device, dtype=torch.int)
        for i, _ in enumerate(random_actions):
            if random_actions[i]:
                actions[i] = torch.randint(action_values.shape[-1], (1,), device=action_values.device)
            else:
                actions[i] = greedy_actions[i]

        return actions

    @classmethod
    def from_dict(cls, data: dict) -> 'EpsilonGreedy':
        """
        Reconstruct the decision function from a dictionary.
        Child classes should override this if they have custom parameters.

        Args:
            data (dict): dictionary containing parameter values

        Returns:
            DecisionFunction: Decision function object
        """
        if data["type"] != cls.__name__:
            raise ValueError(f"Cannot load {data['type']} as {cls.__name__}")
        return cls(epsilon=data["epsilon"])

    def to_dict(self) -> dict:
        """
        Serialize the decision function to a dictionary.
        Child classes should override this if they have custom parameters.

        Returns:
            dict: dictionary containing class type and parameter values
        """
        return {
            "type": self.__class__.__name__,
            "epsilon": self.epsilon
        }


class Softmax(DecisionFunction):
    """
    Soft-max
    """

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
        if action_values.ndim != 2:
            raise ValueError(
                "Expected a 1D tensor for actions, but got a tensor with shape: {}".format(action_values.shape))

        # Compute exponential values scaled by tau
        exp_values = torch.exp(action_values / self.tau)

        # Normalize to get probabilities
        action_pmf = exp_values / torch.sum(exp_values)

        # Sample from the probabilities to get the action
        action = stochastic_selection(action_pmf)

        return action

    @classmethod
    def from_dict(cls, data: dict) -> 'Softmax':
        """
        Reconstruct the decision function from a dictionary.
        Child classes should override this if they have custom parameters.

        Args:
            data (dict): dictionary containing parameter values

        Returns:
            DecisionFunction: Decision function object
        """
        if data["type"] != cls.__name__:
            raise ValueError(f"Cannot load {data['type']} as {cls.__name__}")
        return cls(tau=data["tau"])

    def to_dict(self) -> dict:
        """
        Serialize the decision function to a dictionary.
        Child classes should override this if they have custom parameters.

        Returns:
            dict: dictionary containing class type and parameter values
        """
        return {
            "type": self.__class__.__name__,
            "tau": self.tau
        }


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
