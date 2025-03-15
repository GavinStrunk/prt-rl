from tensordict.tensordict import TensorDict
import threading
import torch
from typing import Dict
from prt_rl.env.interface import EnvParams
from prt_rl.common.policy.policies import Policy

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