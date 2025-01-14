import numpy as np
import os
from typing import Optional, Tuple
from prt_sim.jhu.base import BaseEnvironment
from prt_sim.rendering import GridworldRender


class RobotGame(BaseEnvironment):
    """
    Robot Game is a discrete grid world navigated by Fred the robot.

    .. image:: /_static/robot_game.png
        :alt: Robot Game
        :width: 100%
        :align: center

    **Action space**: integer representing a discrete action described in the table below

    +-----+--------+
    | Num | Action |
    +=====+========+
    | 0   | up     |
    +-----+--------+
    | 1   | down   |
    +-----+--------+
    | 2   | left   |
    +-----+--------+
    | 3   | right  |
    +-----+--------+

    **Observation space**: integer representing the current grid space

    **Reward**: +25 for reaching goal, -25 for falling in pit, -1 for every other location

    Examples:


    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5
    }
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self,
                 render_mode: Optional[str] = "rgb_array",
                 ) -> None:
        self.render_mode = render_mode
        self.grid_width = 4
        self.grid_height = 3
        self.world = np.zeros((self.grid_width, self.grid_height))
        self.current_position = np.array([0, 2])
        self.window_surface = None
        self.clock = None

        self.gridworld_render = GridworldRender(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            window_size=(800, 800),
            render_mode=self.render_mode,
            render_fps=self.metadata['render_fps'],
            agent_icons={
                'fred': os.path.join(os.path.dirname(__file__), 'icons/fred.png'),
                'outlet': os.path.join(os.path.dirname(__file__), 'icons/outlet.png'),
                'pit': os.path.join(os.path.dirname(__file__), 'icons/pit.png'),
                'obstacle': os.path.join(os.path.dirname(__file__), 'icons/obstacle.png'),
            },
            window_title='Fred Robot Game',
            background_color=(198, 236, 254)
        )
        self.agent_positions = {
            'outlet': np.array([3, 0]),
            'pit': np.array([3, 1]),
            'obstacle': np.array([2, 1]),
            'fred': self.current_position,
        }

    def get_number_of_states(self) -> int:
        """
        Returns the number of states in the world.

        Returns:
            int: number of states
        """
        return 11

    def get_number_of_actions(self) -> int:
        """
        Returns the number of actions in the world.

        Returns:
            int: number of actions
        """
        return 4



    def reset(self,
              seed: Optional[int] = None,
              randomize_start: Optional[bool] = False
              ) -> int:
        """
        Resets the world to initial state. If initial_state is None, the robot is initialized to state 8.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            randomize_start (bool, optional): Whether to randomize the starting state. Not all environments will support this. Defaults to False.

        Returns:
            int: current state value
        """
        assert not randomize_start, "Randomizing the start is not supported"
        self.current_position = np.array([0, 2])
        return self.get_state()

    def execute_action(self,
                       action: int
                       ) -> Tuple[int, float, bool]:
        """
        Executes the action and a step of the world.

        Args:
            action (int): robot action to take

        Returns:
            tuple: a tuple of the (state, reward, done)
        """
        done = False
        reward = -1

        # Handle action
        new_position = self.current_position.copy()
        if action == self.UP:
            new_position[1] -= 1
        elif action == self.DOWN:
            new_position[1] += 1
        elif action == self.LEFT:
            new_position[0] -= 1
        elif action == self.RIGHT:
            new_position[0] += 1
        else:
            raise Exception(f"Invalid action {action} must be [0..3]")

        if self._check_if_valid_action(new_position):
            self.current_position = new_position

        # Done States
        if self.get_state() == 3:
            reward = 25
            done = True
        elif self.get_state() == 6:
            reward = -25
            done = True

        return self.get_state(), float(reward), done

    def get_state(self) -> int:
        """
        Returns the current world state, which is the location of the robot.

        Returns:
            int: current location of the robot
        """
        state = self.current_position[1] * self.grid_width + self.current_position[0]
        if state > 5:
            state -= 1
        return state

    def _check_if_valid_action(self, position: list[int]) -> bool:
        """
        Checks if the location of the robot after taking an action is valid.

        Args:
            position (list[int]): location of the robot to check

        Returns:
            bool: True if the location is valid, False otherwise
        """
        # Agent is trying to go off the top or bottom of the grid
        if position[1] < 0 or position[1] >= self.grid_height:
            return False

        # Agent is trying to get off the left or right part of the grid
        if position[0] < 0 or position[0] >= self.grid_width:
            return False

        # This is a blank location on the grid
        if np.array_equal(position, np.array([2, 1])):
            return False

        return True

    def render(self):
        self.agent_positions['fred'] = self.current_position

        if self.render_mode == 'human':
            self.gridworld_render.render(self.agent_positions)
        elif self.render_mode == 'rgb_array':
            return self.gridworld_render.render(self.agent_positions)


if __name__ == '__main__':
    env = RobotGame()
    env.reset()
    env.render()

    for _ in range(20):
        _, _, done = env.execute_action(np.random.randint(4))
        env.render()

        if done:
            break