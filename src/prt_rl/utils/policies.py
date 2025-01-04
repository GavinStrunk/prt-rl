from abc import ABC, abstractmethod
from dataclasses import asdict
import io
from typing import Any, Dict, Optional
import torch
import threading
from tensordict.tensordict import TensorDict
from prt_rl.env.interface import EnvParams
import prt_rl.utils.qtable as qtabs
import prt_rl.utils.networks as qnets
import prt_rl.utils.decision_functions as dfuncs

def load_from_mlflow(
        tracking_uri: str,
        model_name: str,
        model_version: str,
) -> 'Policy':
    """
    Loads a model that is either registered in mlflow or associated with a run id.

    Args:
        tracking_uri (str): mlflow tracking uri
        model_name (str): name of the model in the registry
        model_version (str): string version of the model

    Returns:
        Policy: policy object
    """
    try:
        import mlflow
    except ModuleNotFoundError:
        raise ModuleNotFoundError("mlflow is required to be installed load a policy from mlflow")

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    registered_models = client.search_registered_models()
    for model in registered_models:
        print(f"Model Name: {model.name}")

    model_str = f"models:/{model_name}/{model_version}"
    policy = mlflow.pyfunc.load_model(model_uri=model_str)

    # Extract the metadata
    metadata = policy.metadata.metadata

    # Policy factory
    if metadata['type'] == 'QTablePolicy':
        return QTablePolicy.load_from_dict(metadata['policy'])
    return policy

class Policy(ABC):
    """
    Base class for implementing policies.

    Args:
        env_params (EnvParams): Environment parameters.
        device (str): The device to use.
    """
    def __init__(self,
                 env_params: EnvParams,
                 device: str = 'cpu',
                 ) -> None:
        self.env_params = env_params
        self.device = device

    @abstractmethod
    def get_action(self,
                   state: TensorDict
                   ) -> TensorDict:
        """
        Chooses an action based on the current state. Expects the key "observation" in the state tensordict

        Args:
            state (TensorDict): current state tensordict

        Returns:
            TensorDict: tensordict with the "action" key added
        """
        raise NotImplementedError

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        """
        Sets a key value parameter

        Args:
            name (str): name of the parameter
            value (Any): value to set
        """
        pass


    def save_to_dict(self) -> dict:
        """
        Serializes the policy into a dictionary.
        Child classes should override this method if they are capable of being saved.
        """
        return {}


class RandomPolicy(Policy):
    """
    Implements a policy that uniformly samples random actions.

    Args:
        env_params (EnvParams): environment parameters
    """
    def __init__(self,
                 env_params: EnvParams,
                 ) -> None:
        super(RandomPolicy, self).__init__(env_params=env_params)

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


class KeyboardPolicy(Policy):
    """
    The keyboard policy allows interactive control of the agent using keyboard input.

    Notes:
        I could modify this to implement "sticky" keys, so in non-blocking the last key pressed stays the action until a new key is pressed. Alternatively, you could set a default value and the action goes back to a default when the key is released.

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
        super(KeyboardPolicy, self).__init__(env_params=env_params)
        self.keyboard = keyboard
        self.key_action_map = key_action_map
        self.blocking = blocking
        self.latest_key = None
        self.listener_thread = None

        if not self.blocking:
            self._start_listener()

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
            # Non-blocking: use the latest key press
            key_string = self.latest_key
            if key_string not in self.key_action_map:
                # If no valid key press, use a default action or skip
                action_val = 0  # Example: default or no-op action
            else:
                action_val = self.key_action_map[key_string]
                self.latest_key = None  # Reset the latest key so another key has to be pressed.

        state['action'] = torch.tensor([[action_val]])
        return state

    def _start_listener(self):
        """
        Starts a background thread to listen for key presses.
        """
        def listen_for_keys():
            def on_press(key):
                try:
                    if isinstance(key, self.keyboard.KeyCode):
                        self.latest_key = key.char
                    elif isinstance(key, self.keyboard.Key):
                        self.latest_key = key.name
                except Exception as e:
                    print(f"Error in key press listener: {e}")

            with self.keyboard.Listener(on_press=on_press, suppress=True) as listener:
                listener.join()

        self.listener_thread = threading.Thread(target=listen_for_keys, daemon=True)
        self.listener_thread.start()

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
    """
    A Q-Table policy combines a q-table action value function with a decision function.

    Args:
        env_params (EnvParams): environment parameters
        num_envs (int): number of environments
        decision_function (DecisionFunction): decision function. If None (default), EpsilonGreedy is used with an epsilon of 0.1.
        qtable (QTable, optional): Q-Table. If None (default), Q-Table will be created with initial values of 0 and no visit tracking.
        device (str): String device name. Default is 'cpu'.
    
    """
    def __init__(self,
                 env_params: EnvParams,
                 num_envs: int = 1,
                 decision_function: Optional[dfuncs.DecisionFunction] = None,
                 qtable: Optional[qtabs.QTable] = None,
                 device: str = 'cpu'
                 ):
        super(QTablePolicy, self).__init__(env_params=env_params, device=device)
        assert env_params.action_continuous == False, "QTablePolicy only supports discrete action spaces."
        assert env_params.observation_continuous == False, "QTablePolicy only supports discrete observation spaces."

        self.num_envs = num_envs

        if qtable is None:
            self.q_table = qtabs.QTable(
                    state_dim=self.env_params.observation_max+1,
                    action_dim=self.env_params.action_max+1,
                    batch_size=num_envs,
                    initial_value=0.0,
                    track_visits=False,
                    device=device,
                )
        else:
            self.q_table = qtable

        if decision_function is None:
            self.decision_function = dfuncs.EpsilonGreedy(epsilon=0.1)
        else:
            self.decision_function = decision_function

    def get_action(self,
                   state: TensorDict
                   ) -> TensorDict:
        obs_val = state['observation']
        q_values = self.q_table.get_action_values(obs_val)

        action = self.decision_function.select_action(q_values)
        state['action'] = action
        return state

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        if hasattr(self.decision_function, name):
            self.decision_function.set_parameter(name, value)
        else:
            raise ValueError(f"Parameter '{name}' not found in QTablePolicy.")

    def get_qtable(self) -> qtabs.QTable:
        """
        Returns the Q-Table used in the policy.

        Returns:
            QTable: Q-Table
        """
        return self.q_table

    @classmethod
    def load_from_dict(cls, data: dict) -> 'QTablePolicy':
        """
        Constructs a QTablePolicy from a dictionary. It is assumed the data dictionary was saved with the save_to_dict method.

        Args:
            data (dict): Dictionary of QTablePolicy parameters

        Returns:
            QTablePolicy: Q-Table policy object
        """
        env_params = EnvParams(**data['env_params'])

        # Dynamically load the decision function
        decision_function_class = getattr(dfuncs, data['decision_function']['type'])
        decision_function = decision_function_class.from_dict(data['decision_function'])

        # Deserialize q_table from binary data
        q_table_buffer = io.BytesIO(data["q_table"])
        q_table_data = torch.load(q_table_buffer)

        # Deserialize visit table from binary data if it exists
        if data['visit_table'] is not None:
            visit_table_buffer = io.BytesIO(data['visit_table'])
            visit_table_data = torch.load(visit_table_buffer)
        else:
            visit_table_data = None

        # Dynamically create QTable class and load qtable and visit table
        q_table_class = getattr(qtabs, data['q_table_class'])
        q_table = q_table_class(**data['q_table_init_args'])
        q_table.q_table = q_table_data
        q_table.visit_table = visit_table_data

        # Construct the QTablePolicy
        policy = cls(
            env_params=env_params,
            num_envs=data['num_envs'],
            decision_function=decision_function,
            qtable=q_table,
            device=data['device']
        )
        return policy

    def save_to_dict(self) -> dict:
        """
        Serializes the QTablePolicy into a dictionary to it can be saved.

        Returns:
            dict: Dictionary of QTablePolicy parameters and values needed to load it.
        """
        # Serialize the q_table tensor to binary data
        q_table_buffer = io.BytesIO()
        torch.save(self.q_table.q_table, q_table_buffer)
        q_table_buffer.seek(0)

        # Serialize the visit table tensor to binary data if there is one
        visit_table = None
        if self.q_table.track_visits:
            visit_table_buffer = io.BytesIO()
            torch.save(self.q_table.visit_table, visit_table_buffer)
            visit_table_buffer.seek(0)
            visit_table = visit_table_buffer.getvalue()

        return {
            'env_params': asdict(self.env_params),
            'num_envs': self.num_envs,
            'decision_function': self.decision_function.to_dict(),
            'q_table_class': self.q_table.__class__.__name__,
            'q_table_init_args': self.q_table.init_args(),
            'q_table': q_table_buffer.getvalue(),
            'visit_table': visit_table,
            'device': self.device,
        }

class QNetworkPolicy(Policy):
    """
    QNetwork policy is an ANN based q value function approximation.

    Args:
        env_params (EnvParams): environment parameters
        num_envs (int): number of environments
        decision_function (DecisionFunction): decision function. If None (default), EpsilonGreedy is used with an epsilon of 0.1.
        qnetwork (torch.nn.Sequential, optional): QNetwork. If None (default), an MLP QNetwork will be created.
        device (str): String device name. Default is 'cpu'.
    """
    def __init__(self,
                 env_params: EnvParams,
                 num_envs: int = 1,
                 decision_function: Optional[dfuncs.DecisionFunction] = None,
                 qnetwork: Optional[torch.nn.Sequential] = None,
                 device: str = 'cpu'
                 ) -> None:
        super(QNetworkPolicy, self).__init__(env_params=env_params, device=device)
        self.num_envs = num_envs

        if qnetwork is None:
            self.q_network = qnets.MLP(
                state_dim=self.env_params.observation_max+1,
                action_dim=self.env_params.action_max+1,
            )
        else:
            self.q_network = qnetwork

        if decision_function is None:
            self.decision_function = dfuncs.EpsilonGreedy(epsilon=0.1)
        else:
            self.decision_function = decision_function

    def get_action(self,
                   state: TensorDict
                   ) -> TensorDict:
        state = state['observation']
        q_values = self.q_network.get_action(state)
        action = self.decision_function.select_action(q_values)
        state['action'] = action
        return state

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        if hasattr(self.decision_function, name):
            self.decision_function.set_parameter(name, value)
        else:
            raise ValueError(f"Parameter '{name}' not found in QNetworkPolicy.")

    @classmethod
    def load_from_dict(cls, data: dict) -> 'QNetworkPolicy':
        """
        Constructs a QNetworkPolicy from a dictionary. It is assumed the data dictionary was saved with the save_to_dict method.

        Args:
            data (dict): Dictionary of QNetworkPolicy parameters

        Returns:
            QNetworkPolicy: QNetwork policy object
        """
        env_params = EnvParams(**data['env_params'])

        # Dynamically load the decision function
        decision_function_class = getattr(dfuncs, data['decision_function']['type'])
        decision_function = decision_function_class.from_dict(data['decision_function'])

        # Load the QNetwork dynamically
        q_network_class = getattr(qnets, data['q_network_class'])
        q_network = q_network_class(**data['q_network_init_args'])
        q_network.load_state_dict(data['q_network'])

        policy = cls(
            env_params=env_params,
            num_envs=data['num_envs'],
            decision_function=decision_function,
            qnetwork=q_network,
            device=data['device']
        )
        return policy

    def save_to_dict(self) -> dict:
        """
        Serializes the QNetworkPolicy into a dictionary to it can be saved.

        Returns:
            dict: Dictionary of QNetworkPolicy parameters and values needed to load it.
        """
        return {
            'env_params': asdict(self.env_params),
            'num_envs': self.num_envs,
            'decision_function': self.decision_function.to_dict(),
            'q_network_class': self.q_network.__class__.__name__,
            'q_network_init_args': self.q_network.init_args(),
            'q_network': self.q_network.state_dict(),
            'device': self.device,
        }
