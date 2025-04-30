from abc import ABC, abstractmethod
import numpy as np
import pygame


class Visualizer(ABC):
    def start(self):
        pass

    def stop(self):
        pass

    def show(self, frame: np.ndarray) -> None:
        pass

class PygameVisualizer(Visualizer):
    def __init__(self,
                 fps: int = 50,
                 caption: str = 'Visualizer',
                 ) -> None:
        self.fps = fps
        self.caption = caption
        self.clock = None
        self.window_size = None
        self.screen = None

    def start(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(self.caption)
        self.clock = pygame.time.Clock()

    def stop(self):
        pygame.quit()

    def show(self, frame: np.ndarray) -> None:
        if self.window_size is None:
            height, width, _ = frame.shape
            self.window_size = (width, height)
            self.screen = pygame.display.set_mode(self.window_size)

        # If the frame is grayscale (H, W, 1), convert it to (H, W)
        if frame.shape[-1] == 1:
            frame = frame[:, :, 0]

        # Make a surface from the RGB array
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        # Blit the surface onto the screen
        self.screen.blit(surface, (0, 0))

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.fps)
