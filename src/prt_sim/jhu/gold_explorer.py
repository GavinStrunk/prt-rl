import numpy as np
import random
import os
from typing import Tuple, Optional
from prt_sim.jhu.base import BaseEnvironment
from prt_sim.rendering import GridworldRender


def get_state_index(x, y, z):
    x_idx = x
    y_idx = 8 * y
    z_idx = 64 * z
    return x_idx + y_idx + z_idx   # ranges from 0 to 127


class GoldExplorer(BaseEnvironment):
    """
    The Gold Explorer puzzle

    .. image:: /_static/gold-explorer.png
       :alt: Gold Explorer puzzle
       :width: 100%
       :align: center

    **Action space**: integer representing a discrete action described in the table below

    +-----+--------+
    | Num | Action |
    +=====+========+
    | 0   | North  |
    +-----+--------+
    | 1   | East   |
    +-----+--------+
    | 2   | South  |
    +-----+--------+
    | 3   | West   |
    +-----+--------+

    **Observation space**: integer between 0 and 127 representing the state as an octal number, <gold bit><row><column>

    .. image:: /_static/gold-explorer-state.png
        :alt: Gold Explorer State
        :width: 100%
        :align: center

    **Reward**: +15 for obtaining gold coins, +30 for obtaining the motherlode, -30 for entering a mine field, -1 for every other location
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5
    }

    def __init__(self,
                 render_mode: Optional[str] = "rgb_array"
                 ) -> None:
        self.render_mode = render_mode
        self.num_states = 128
        self.num_actions = 4
        self.expl_x = 0  # explorer's x position from 0 to 7
        self.expl_y = 0  # explorer's y position from 0 to 7
        self.expl_z = 0  # explorer's z position from 0 to 1
        self.win = {15, 62, 79, 126}
        self.loss = {13, 17, 28, 32, 51, 77, 81, 92, 96, 115}
        self.coins = {50}
        self.mount = {1, 9, 49, 65, 73, 113}

        self.gridworld_render = GridworldRender(
            grid_width=8,
            grid_height=8,
            window_size=(800, 800),
            render_mode=self.render_mode,
            render_fps=self.metadata['render_fps'],
            agent_icons={
                'explorer': os.path.join(os.path.dirname(__file__), 'icons/explorer.png'),
                'mountain1': os.path.join(os.path.dirname(__file__), 'icons/mountain.png'),
                'mountain2': os.path.join(os.path.dirname(__file__), 'icons/mountain.png'),
                'mountain3': os.path.join(os.path.dirname(__file__), 'icons/mountain.png'),
                'mine1': os.path.join(os.path.dirname(__file__), 'icons/mine.png'),
                'mine2': os.path.join(os.path.dirname(__file__), 'icons/mine.png'),
                'mine3': os.path.join(os.path.dirname(__file__), 'icons/mine.png'),
                'mine4': os.path.join(os.path.dirname(__file__), 'icons/mine.png'),
                'mine5': os.path.join(os.path.dirname(__file__), 'icons/mine.png'),
                'coins': os.path.join(os.path.dirname(__file__), 'icons/coins.png'),
                'gold1': os.path.join(os.path.dirname(__file__), 'icons/gold.png'),
                'gold2': os.path.join(os.path.dirname(__file__), 'icons/gold.png'),
            },
            window_title='Gold Explorer',
            background_color=(198, 236, 254)
        )
        self.agent_positions = {
            'mountain1': np.array([1, 0]),
            'mountain2': np.array([1, 1]),
            'mountain3': np.array([1, 6]),
            'mine1': np.array([1, 2]),
            'mine2': np.array([0, 4]),
            'mine3': np.array([5, 1]),
            'mine4': np.array([4, 3]),
            'mine5': np.array([3, 6]),
            'coins': np.array([2, 6]),
            'gold1': np.array([7, 1]),
            'gold2': np.array([6, 7]),
            'explorer': np.array([0, 0]),
        }

    # Get the key environment parameters
    def get_number_of_states(self) -> int:
        """
        Returns the number of states in the puzzle

        Returns:
            int: total number of states in the puzzle
        """
        return self.num_states

    def get_number_of_actions(self) -> int:
        """
        Returns the number of discrete actions in the puzzle

        Returns:
            int: total number of actions in the puzzle
        """
        return self.num_actions

    # Get the state IDs that should not be set optimistically
    def get_terminal_states(self):
        term = self.win.union(self.loss, self.mount)
        return term

    def get_state(self):
        return get_state_index(self.expl_x, self.expl_y, self.expl_z)

    # Set the current state to the initial state
    def reset(self,
              seed: Optional[int] = None,
              randomize_start: Optional[bool] = False
              ) -> int:
        x = 0
        y = 0
        z = 0
        if randomize_start:
            done = False
            while not done:
                x = random.randint(0, 7)
                y = random.randint(0, 7)
                z = random.randint(0, 1)
                st = get_state_index(x, y, z)
                if (st in self.win) or (st in self.loss) or (st in self.mount) or (st in self.coins):
                    done = False
                else:
                    done = True
        self.expl_x = x
        self.expl_y = y
        self.expl_z = z
        st = get_state_index(self.expl_x, self.expl_y, self.expl_z)
        return st

    def execute_action(self,
                       action: int
                       ) -> Tuple[int, float, bool]:
        """
        Executes an action for the explorer.

        Args:
            action (int): the action to execute

        Returns:

        """
        # Use the agent's action to determine the next state and reward #
        # Note: 'N' = 0, 'E' = 1, 'S' = 2, 'W' = 3 #

        current_state = get_state_index(self.expl_x, self.expl_y, self.expl_z)
        new_state = current_state
        reward = 0
        game_end = False

        # if in terminal states, stay in terminal states
        if (current_state in self.win) or (current_state in self.loss):
            new_state = current_state
            reward = 0
            game_end = True

        elif (current_state in self.mount) or (current_state in self.coins):
            new_state = current_state
            reward = -1000
            game_end = True

        else:
            temp_x = self.expl_x
            temp_y = self.expl_y
            temp_z = self.expl_z

            # determine a potential next state
            if action == 0:  # action is 'N'
                if temp_y == 0:
                    temp_y = 0
                else:
                    temp_y = temp_y - 1

            elif action == 1:  # action is 'E'
                if temp_x == 7:
                    temp_x = 7
                else:
                    temp_x = temp_x + 1

            elif action == 2:  # action is 'S'
                if temp_y == 7:
                    temp_y = 7
                else:
                    temp_y = temp_y + 1

            else:  # action is 'W'
                if temp_x == 0:
                    temp_x = 0
                else:
                    temp_x = temp_x - 1

            # recalculate the new state
            new_state = get_state_index(temp_x, temp_y, temp_z)

            # check to see if coins can be picked up
            if new_state in self.coins:
                temp_z = 1  # shift to second level grid space
                new_state = get_state_index(temp_x, temp_y, temp_z)
                reward = 15
                game_end = False

            elif new_state in self.mount:
                temp_x = self.expl_x
                temp_y = self.expl_y
                temp_z = self.expl_z
                new_state = get_state_index(temp_x, temp_y, temp_z)
                reward = -1
                game_end = False

            elif new_state in self.loss:      # you lose
                reward = -30
                game_end = True

            elif new_state in self.win:     # you won
                reward = 30
                game_end = True

            else:
                reward = -1
                game_end = False

            self.expl_x = temp_x
            self.expl_y = temp_y
            self.expl_z = temp_z

        return new_state, reward, game_end

    def render(self):
        self.agent_positions['explorer'] = np.array([self.expl_x, self.expl_y])
        self.agent_positions['coins'] = None if self.expl_z == 1 else self.agent_positions['coins']
        
        if self.render_mode == 'human':
            self.gridworld_render.render(self.agent_positions)
        elif self.render_mode == 'rgb_array':
            return self.gridworld_render.render(self.agent_positions)


if __name__ == '__main__':
    env = GoldExplorer()
    env.reset(randomize_start=False)
    env.render()

    for _ in range(20):
        _, _, done = env.execute_action(np.random.randint(env.num_actions))
        env.render()

        if done:
            break