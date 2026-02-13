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

    @staticmethod
    def _normalize_action_values(action_values: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """
        Normalize action values to batched shape (N, A).

        Returns:
            tuple[Tensor, bool]: (normalized_tensor, was_unbatched)
        """
        if action_values.ndim == 1:
            return action_values.unsqueeze(0), True
        if action_values.ndim == 2:
            return action_values, False
        raise ValueError(
            "Expected action_values with shape (# env, # actions) or (# actions,), "
            f"but got shape: {tuple(action_values.shape)}"
        )

    @staticmethod
    def _restore_action_shape(actions: torch.Tensor, was_unbatched: bool) -> torch.Tensor:
        """
        Restore output shape for single-state inputs.
        """
        if was_unbatched:
            return actions.squeeze(0)
        return actions

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        """
        Sets a named parameter in the decision function.

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
        action_values, was_unbatched = self._normalize_action_values(action_values)

        # Random tie-break across argmax actions.
        max_value = torch.max(action_values, dim=1, keepdim=True).values
        max_mask = action_values == max_value

        random_scores = torch.rand_like(action_values)
        random_scores = random_scores.masked_fill(~max_mask, -1.0)
        chosen_indices = torch.argmax(random_scores, dim=1, keepdim=True).to(dtype=torch.long)

        return self._restore_action_shape(chosen_indices, was_unbatched)


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
        action_values, was_unbatched = self._normalize_action_values(action_values)

        # Greedy action selection
        greedy_actions = super().select_action(action_values).to(dtype=torch.long)

        # Epsilon-greedy logic
        random_mask = torch.rand((action_values.size(0), 1), device=action_values.device) <= self.epsilon
        random_actions = torch.randint(
            low=0,
            high=action_values.shape[-1],
            size=(action_values.shape[0], 1),
            device=action_values.device,
            dtype=torch.long,
        )
        actions = torch.where(random_mask, random_actions, greedy_actions)

        return self._restore_action_shape(actions, was_unbatched)

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
        action_values, was_unbatched = self._normalize_action_values(action_values)
        if self.tau <= 0:
            raise ValueError("Temperature 'tau' must be > 0.")

        # Compute exponential values scaled by tau (stabilized).
        centered = action_values - torch.max(action_values, dim=1, keepdim=True).values
        exp_values = torch.exp(centered / self.tau)

        # Normalize to get probabilities
        action_pmf = exp_values / torch.sum(exp_values, dim=1, keepdim=True)

        # Sample from the probabilities to get the action
        action = stochastic_selection(action_pmf)

        return self._restore_action_shape(action, was_unbatched)

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

    def select_action(self,
                      action_values: torch.Tensor,
                      action_selections: torch.Tensor | None = None
                      ) -> torch.Tensor:
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
        action_values, was_unbatched = self._normalize_action_values(action_values)

        if self.c <= 0:
            raise ValueError("The constant 'c' must be greater than 0.")
        if self.t <= 0:
            raise ValueError("The timestep 't' must be greater than 0.")

        if action_selections is None:
            # Fall back to greedy selection when counts are not provided.
            greedy_actions = Greedy().select_action(action_values)
            return self._restore_action_shape(greedy_actions, was_unbatched)

        action_selections, _ = self._normalize_action_values(action_selections)
        if action_values.shape != action_selections.shape:
            raise ValueError("Actions and action_selections must have the same shape.")
        if (action_selections <= 0).any():
            raise ValueError("Action selection counts must be > 0.")

        # Compute UCB values
        log_term = torch.log(torch.tensor(self.t, dtype=torch.float32, device=action_values.device))
        exploration_bonus = self.c * torch.sqrt(log_term / action_selections)
        ucb_values = action_values + exploration_bonus

        # Random tie-break across argmax actions.
        max_value = torch.max(ucb_values, dim=1, keepdim=True).values
        max_mask = ucb_values == max_value
        random_scores = torch.rand_like(ucb_values)
        random_scores = random_scores.masked_fill(~max_mask, -1.0)
        selected_action = torch.argmax(random_scores, dim=1, keepdim=True).to(dtype=torch.long)

        return self._restore_action_shape(selected_action, was_unbatched)

    @classmethod
    def from_dict(cls, data: dict) -> "UpperConfidenceBound":
        if data["type"] != cls.__name__:
            raise ValueError(f"Cannot load {data['type']} as {cls.__name__}")
        return cls(c=data["c"], t=data["t"])

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "c": self.c,
            "t": self.t,
        }
