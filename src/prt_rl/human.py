from enum import Enum, auto
import threading
import torch
from typing import Dict, Tuple, Union
from prt_rl.env.interface import EnvParams
from prt_rl.agent import BaseAgent

class GameControllerAgent(BaseAgent):
    """
    The game controller policy allows interactive control of an agent with discrete or continuous actions.

    For continuous actions, the key_action_map maps a game controller input to an action index rather than a value. For example, 'JOY_RIGHT_X': 1 would map the x direction of the right joystick to action index 1.
    Notes:
        You don't want to use blocking with continuous actions because this would result in jerky agents.
        I also need to consider half joystick moves. For example if th input is speed you don't want to hold a joystick down to go nowhere.
        I should accept a default value for the actions as well.

    Args:
        env_params (EnvParams): environment parameters
        key_action_map : mapping from key string to action value
        blocking (bool, optional): Whether the policy blocks at each step. Defaults to True.

    Raises:
        ImportError: If inputs is not installed.
        AssertionError: If a game controller is not found
    """
    class Key(Enum):
        BUTTON_A = auto()
        BUTTON_B = auto()
        BUTTON_X = auto()
        BUTTON_Y = auto()
        BUTTON_LB = auto()
        BUTTON_LT = auto()
        BUTTON_RB = auto()
        BUTTON_RT = auto()
        BUTTON_START = auto()
        BUTTON_BACK = auto()
        BUTTON_DPAD_UP = auto()
        BUTTON_DPAD_DOWN = auto()
        BUTTON_DPAD_LEFT = auto()
        BUTTON_DPAD_RIGHT = auto()
        BUTTON_JOY_RIGHT = auto()
        BUTTON_JOY_LEFT = auto()
        JOYSTICK_LEFT_X = auto()
        JOYSTICK_LEFT_Y = auto()
        JOYSTICK_RIGHT_X = auto()
        JOYSTICK_RIGHT_Y = auto()
        # JOYSTICK_LT = auto()
        # JOYSTICK_RT = auto()

    EVENT_TYPE_TO_KEY_MAP = {
        'BTN_THUMB': Key.BUTTON_A,
        'BTN_THUMB2': Key.BUTTON_B,
        'BTN_TRIGGER': Key.BUTTON_X,
        'BTN_TOP': Key.BUTTON_Y,
        'BTN_TOP2': Key.BUTTON_LB,
        'BTN_BASE': Key.BUTTON_LT,
        'BTN_PINKIE': Key.BUTTON_RB,
        'BTN_BASE2': Key.BUTTON_RT,
        'BTN_BASE4': Key.BUTTON_START,
        'BTN_BASE3': Key.BUTTON_BACK,
        'BTN_DPAD_UP': Key.BUTTON_DPAD_UP,
        'BTN_DPAD_DOWN': Key.BUTTON_DPAD_DOWN,
        'BTN_DPAD_LEFT': Key.BUTTON_DPAD_LEFT,
        'BTN_DPAD_RIGHT': Key.BUTTON_DPAD_RIGHT,
        'BTN_BASE5': Key.BUTTON_JOY_LEFT,
        'BTN_BASE6': Key.BUTTON_JOY_RIGHT,
        'ABS_X': Key.JOYSTICK_LEFT_X,
        'ABS_Y': Key.JOYSTICK_LEFT_Y,
        'ABS_Z': Key.JOYSTICK_RIGHT_X,
        'ABS_RZ': Key.JOYSTICK_RIGHT_Y,
    }
    def __init__(self,
                 key_action_map: Union[Dict[Key, int], Dict[Key, int | str]],
                 blocking: bool = True,
                 ) -> None:
        try:
            import inputs
            self.inputs = inputs
        except ImportError as e:
            raise ImportError(
                "The 'inputs' library is required for GameController but is not installed. "
                "Please install it using 'pip install inputs'."
            ) from e
        super(GameControllerAgent, self).__init__(policy=None)
        self.key_action_map = key_action_map
        self.blocking = blocking
        self.continuous = self.env_params.action_continuous
        self.joystick_min = -1.0
        self.joystick_max = 1.0

        # Check if a game controller is found
        gamepads = self.inputs.devices.gamepads
        if not gamepads:
            raise AssertionError("No game controller found")

        # Start read thread if the policy is non-blocking
        if not self.blocking:
            self.listener_thread = None
            self.running = False
            self.latest_values = torch.zeros(*self.env_params.action_shape)
            self.lock = threading.Lock()
            self._start_listener()

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Gets a game controller input and maps it to the action space.

        Args:
            state (TensorDict): A tensor representing the current state of the environment.

        Returns:
            A TensorDict with the "action" key added.
        """
        assert state.batch_size[0] == 1, "GameController only supports batch size 1 for now."

        # Get the data type for the action values
        if self.continuous:
            ttype = torch.float32
        else:
            ttype = torch.int

        if self.blocking:
            key = None
            while key not in self.key_action_map:
                key = self.EVENT_TYPE_TO_KEY_MAP[self._wait_for_inputs()]

            action = self.key_action_map[key]
            if isinstance(action, int):
                action_val = torch.tensor([self.key_action_map[key]], dtype=ttype)
            else:
                raise ValueError(f"Unsupported action {action}")
        else:
            # Non-blocking: use the latest key presses
            # Grab the latest values and updated current
            # Scale the joystick value to the action range for continuous
            with self.lock:
                action_val = self.latest_values.clone()

        state['action'] = action_val.unsqueeze(0)
        return state

    def train(self, env, total_frames, schedulers = None, logger = None, logging_freq = 1000):
        raise NotImplementedError("GameControllerAgent does not support training. It is designed for interactive control only.")

    def _start_listener(self):
        """
        Starts an event listening thread that captures game controller inputs and updates the latest action values.
        """
        self.running = True
        def event_loop():
            while self.running:
                events = self.inputs.get_gamepad()
                for event in events:
                    match event.ev_type:
                        case "Key":
                            pass
                        case "Absolute":
                            joy_name = event.code
                            joy_value = event.state

                            # Convert the joystick name to a Key
                            if joy_name in self.EVENT_TYPE_TO_KEY_MAP.keys():
                                joy_key = self.EVENT_TYPE_TO_KEY_MAP[joy_name]

                                # Get the action map from the Key
                                if joy_key in self.key_action_map.keys():
                                    action_map = self.key_action_map[joy_key]

                                    # Normalize joystick value to [-1, 1]
                                    # Note: the X direction neutral value is 127, but the Y direction neutral value is 128
                                    if joy_name == 'ABS_X' or joy_name == 'ABS_Z':
                                        norm_joy_value = (joy_value - 127.0) / 127.0
                                    else:
                                        norm_joy_value = -(joy_value - 128.0) / 128.0

                                    # Process action and value
                                    self._process_joystick(action_map, norm_joy_value)
                        case "Misc":
                            # Ignore MISC messages
                            pass
                        case "Sync":
                            # Ignore Sync messages
                            pass
                        case _:
                            print(f"Unknown key: {event.ev_type}")

        self.listener_thread = threading.Thread(target=event_loop, daemon=True)
        self.listener_thread.start()

    def _process_joystick(self,
                          action_map_values: Union[int, Tuple[int, str]],
                          norm_value: float,
                          ) -> None:
        """
        Updates the latest action values using the normalized joystick value based on the action map parameters.

        Args:
            action_map_values (Union[int, Tuple[int, str]]): Action map values.
            norm_value (float): Normalized joystick value between [-1, 1]
        """
        if isinstance(action_map_values, int):
            action_index = action_map_values
            action_param = None
        elif len(action_map_values) == 2:
            action_index, action_param = action_map_values
        else:
            raise ValueError(f"Unsupported action map {action_map_values}")

        # Normalized joystick value ranges from [-1.0, 1.0]
        joystick_min = -1.0
        joystick_max = 1.0

        if action_param == 'positive':
            # Change the joystick range from [0, 1] and clip the negative norm values to 0
            joystick_min = 0.0
            norm_value = max(joystick_min, norm_value)

        if action_param == 'negative':
            # Change the joystick range from [-1.0, 0] and clip the positive norm value to 0
            joystick_max = 0.0
            norm_value = min(joystick_max, norm_value)

        # Get the action min/max for this index and scale action value from [joy_min, joy_max] to [action_min, action_max]
        action_min = self.env_params.action_min[action_index]
        action_max = self.env_params.action_max[action_index]
        action_value = ((norm_value - joystick_min) / (joystick_max - joystick_min)) * (action_max - action_min) + action_min

        # Update latest action values
        with self.lock:
            self.latest_values[action_index] = action_value

    def _wait_for_inputs(self) -> str:
        """
        Blocking listener that captures a single event and returns the value.

        Returns:
            String value of the Key pressed.
        """
        assert not self.continuous, "Blocking GameController only supports discrete actions."
        key_val = None
        while key_val is None:
            events = self.inputs.get_gamepad()
            for event in events:
                match event.ev_type:
                    case "Key":
                        # Only return the action when the key is pressed and ignore the release
                        if event.state == 1:
                            key_val = event.code
                    case "Absolute":
                        # Read the DPAD buttons
                        print(f"Code: {event.code}  State: {event.state}")
                        if event.code == 'ABS_HAT0X':
                            if event.state == 1:
                                key_val = "BTN_DPAD_RIGHT"
                            if event.state == -1:
                                key_val = "BTN_DPAD_LEFT"

                        if event.code == 'ABS_HAT0Y':
                            if event.state == 1:
                                key_val = "BTN_DPAD_DOWN"
                            if event.state == -1:
                                key_val = "BTN_DPAD_UP"
                        pass
                    case "Misc":
                        # Ignore MISC messages
                        pass
                    case "Sync":
                        # Ignore Sync messages
                        pass
                    case _:
                        print(f"Unknown key: {event.ev_type}")

        return key_val
    
class KeyboardAgent(BaseAgent):
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
        super(KeyboardAgent, self).__init__(policy=None)
        self.keyboard = keyboard
        self.key_action_map = key_action_map
        self.blocking = blocking
        self.latest_key = None
        self.listener_thread = None

        if not self.blocking:
            self._start_listener()

    def predict(self,
                   state: torch.Tensor
                   ) -> torch.Tensor:
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
    
    def train(self, env, total_frames, schedulers = None, logger = None, logging_freq = 1000):
        raise NotImplementedError("KeyboardAgent does not support training. It is designed for interactive control only.")

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