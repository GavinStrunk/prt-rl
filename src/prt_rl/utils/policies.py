from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from prt_rl.env.interface import EnvParams
from prt_rl.utils.qtable import QTable
from prt_rl.utils.decision_functions import DecisionFunction


class Policy(ABC):
    """
    Base class for implementing policies.

    Args:
        decision_function (DecisionFunction): The decision function to use.
        device (str): The device to use.
    """
    def __init__(self,
                 decision_function: DecisionFunction,
                 device: str = 'cpu',
                 ) -> None:
        self.decision_function = decision_function
        self.device = device

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

    # @staticmethod
    # def load_from_file(filename: str) -> 'Policy':
    #     raise NotImplementedError
    #
    # def save(self, filename: str):
    #     raise NotImplementedError

class RandomPolicy:
    """
    Implements a policy that uniformly samples random actions.

    Args:
        env_params (EnvParams): environment parameters
    """
    def __init__(self,
                 env_params: EnvParams,
                 ) -> None:
        self.env_params = env_params

    def get_action(self,
                   state: TensorDict
                   ) -> TensorDict:
        """
        Randomly samples an action from action space.

        Returns:
            TensorDict: Tensordict with the "action" key added
        """
        if not self.env_params.action_continuous:
            # Add 1 to the high value because randint samples between low and 1 less than the high: [low,high)
            action = torch.randint(low=self.env_params.action_min, high=self.env_params.action_max + 1,
                                   size=(*state.batch_size, *self.env_params.action_shape))
        else:
            action = torch.rand(size=(*state.batch_size, *self.env_params.action_shape))
            action = action * (self.env_params.action_max - self.env_params.action_min) + self.env_params.action_min

        state['action'] = action
        return state


class KeyboardPolicy:
    """
    The keyboard policy allows interactive control of the agent using keyboard input.

    Args:
        env_params (EnvParams): environment parameters
        key_action_map (Dict[str, int]): mapping from key string to action value
        blocking (bool): If blocking is True the simulation will wait for keyboard input at each step (synchronous), otherwise the simulation will not block and use the most up-to-date value (asynchronous). Default is True.

    Example:
        from prt_rl.utils.policies import KeyboardPolicy

        policy = KeyboardPolicy(
            env_params=env.get_parameters(),
            key_action_map={
                'up': 0,
                'down': 1,
                'left': 2,
                'right': 3,
            },
            blocking=True
        )

        action_td = policy.get_action(state_td)

    """
    def __init__(self,
                 env_params: EnvParams,
                 key_action_map: Dict[str, int],
                 blocking: bool = True,
                 ) -> None:
        # Check if pynput is installed
        try:
            from pynput import keyboard
        except ImportError as e:
            raise ImportError(
                "The 'pynput' library is required for KeyboardPolicy but is not installed. "
                "Please install it using 'pip install pynput'."
            ) from e
        self.keyboard = keyboard

        self.env_params = env_params
        self.key_action_map = key_action_map
        self.blocking = blocking

    def get_action(self,
                   state: TensorDict
                   ) -> TensorDict:
        """
        Gets a keyboard press and maps it to the action space.
        """
        assert state.batch_size[0] == 1, "KeyboardPolicy Only supports batch size 1 for now."

        if self.blocking:
            key_string = ''
            # Keep reading keys until a valid key in the map is received
            while key_string not in self.key_action_map:
                key_string = self._wait_for_key_press()
            action_val = self.key_action_map[key_string]
        else:
            raise NotImplementedError("Only blocking actions are currently supported in the keyboard policy.")

        state['action'] = torch.tensor([[action_val]])
        return state

    def _wait_for_key_press(self) -> str:
        """
        Blocking method to wait for keyboard press.

        Returns:
            str: String name of the pressed key
        """
        # A callback function to handle key presses
        def on_press(key):
            nonlocal key_pressed
            key_pressed = key  # Store the pressed key
            return False  # Stop the listener after a key is pressed

        key_pressed = None
        # Start the listener in blocking mode
        # Supressing keys keeps them from being passed on to the rest of the computer
        with self.keyboard.Listener(on_press=on_press, suppress=True) as listener:
            listener.join()

        # Get string value of KeyCodes and special Keys
        if isinstance(key_pressed, self.keyboard.KeyCode):
            key_pressed = key_pressed.char
        elif isinstance(key_pressed, self.keyboard.Key):
            key_pressed = key_pressed.name
        else:
            raise ValueError(f"Unrecognized key pressed type: {type(key_pressed)}")

        return key_pressed




class QTablePolicy(Policy):
    def __init__(self,
                 q_table: QTable,
                 decision_function: DecisionFunction,
                 ):
        self.q_table = q_table
        self.decision_function = decision_function

    def get_action(self,
                   state: torch.Tensor
                   ) -> torch.Tensor:
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