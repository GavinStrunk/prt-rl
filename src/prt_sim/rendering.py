import numpy as np
import os
import pygame
from typing import Any, Dict, Tuple


class GridworldRender:
    """
    This class renders a grid world using icons for the agents in the world.

    It is assumed that position (0, 0) is the top left of the window, and x is positive right and y is positive down.

    Args:
        grid_width (int): width of the grid
        grid_height (int): height of the grid
        window_size (Tuple[int, int]): size of the window in pixels
        agent_icons (Dict[str, str]): dictionary of agent names and icon file locations
        render_fps (int): frames per second for rendering
        window_title (str): title of the window

    Examples:

    """
    def __init__(self,
                 grid_width: int,
                 grid_height: int,
                 window_size: Tuple[int, int],
                 agent_icons: Dict[str, str],
                 render_mode: str = "human",
                 render_fps: int = 5,
                 window_title: str = "Gridworld",
                 background_color: Tuple[int, int, int] = (255, 255, 255),
                 ) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.window_size = window_size
        self.agent_icons = self._load_icons(agent_icons)
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.window_title = window_title
        self.background_color = background_color

        self.cell_size = (self.window_size[0] // self.grid_width, self.window_size[1] // self.grid_height)
        self.window_surface = None
        self.clock = None

    def close(self) -> None:
        """
        Safely closes pygame and the display.

        """
        if self.window_surface is not None:
            pygame.quit()

    def render(self,
              agent_positions: Dict[str, np.ndarray],
               ):
        """
        Renders the grid world and the agent icons from dictionary of agent names and positions.

        Args:
            agent_positions (Dict[str, np.ndarray]): dictionary of agent names and numpy array of (x,y) grid positions

        """
        if self.window_surface is None:
            if self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                pygame.display.set_caption(self.window_title)
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._draw_grid()
        self._draw_agent_icons(agent_positions)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.render_fps)
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _load_icons(self, agent_icons: Dict[str, str]) -> Dict[str, Any]:
        """
        Loads icons from the dictionary of agent names and icon filenames.

        Args:
            agent_icons (Dict[str, str]): dictionary of agent names and icon file locations

        Returns:
            Dict[str, Any]: dictionary of agent names and pygame Surfaces
        """
        loaded_icons = {}
        for agent, icon in agent_icons.items():
            icon_file = os.path.join(os.path.dirname(__file__), icon)
            loaded_icons[agent] = pygame.image.load(icon_file)

        return loaded_icons

    def _draw_grid(self) -> None:
        """
        Draws the grid world boundary and cells
        """
        # Draw white rectangular background
        self.window_surface.fill(self.background_color)
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)
                pygame.draw.rect(self.window_surface, (0, 0, 0), rect, 2)

    def _draw_agent_icons(self, agent_positions: Dict[str, np.ndarray]) -> None:
        """
        Scales the agent icon to match the cell size and draws them in the grid world.

        Args:
            agent_positions (Dict[str, np.ndarray]): dictionary of agent names and numpy array of (x,y) grid positions

        """
        for agent, position in agent_positions.items():
            if position is not None:
                x = position[0]
                y = position[1]
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                img = pygame.transform.scale(self.agent_icons[agent], self.cell_size)
                self.window_surface.blit(img, pos)