from abc import ABC, abstractmethod
import imageio
import numpy as np


class Recorder(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def capture_frame(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass


class GifRecorder(Recorder):
    """
    Captures rgb_array data and creates a gif.

    Args:
        fps (int): frames per second
        loop (bool): Whether to loop the GIF after it runs. Defaults to True.
    """

    def __init__(self,
                 fps: int = 10,
                 loop: bool = True
                 ) -> None:
        self.fps = fps
        self.loop = loop
        self.frames = []

    def reset(self):
        """
        Resets the buffer of frames
        """
        self.frames = []

    def capture_frame(self,
                      frame: np.ndarray,
                      ) -> None:
        """
        Captures a frame to be saved to the GIF.

        Args:
            frame (np.ndarray): Numpy rgb array to be saved with format (H, W, C)
        """
        # Ensure the frame is in the correct format (H, W, C)
        if frame.ndim == 2:  # If the frame is grayscale
            frame = np.stack([frame] * 3, axis=-1)
        self.frames.append(frame)

    def save(self,
             filename: str,
             ) -> None:
        """
        Saves the captured frames as a GIF.

        Args:
            filename (str): filename to save GIF to
        """
        if self.loop:
            num_loops = 0
        else:
            num_loops = 1
        imageio.mimsave(filename, self.frames, fps=self.fps, loop=num_loops)
