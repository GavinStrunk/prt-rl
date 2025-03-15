from abc import ABC, abstractmethod
import imageio
import numpy as np


class Recorder(ABC):
    def reset(self) -> None:
        pass

    def capture_frame(self, frame: np.ndarray) -> None:
        pass

    def save(self) -> None:
        pass


class GifRecorder(Recorder):
    """
    Captures rgb_array data and creates a gif.

    Args:
        filename (str): Filename to save the gif.
        fps (int): frames per second
        loop (bool): Whether to loop the GIF after it runs. Defaults to True.
    """

    def __init__(self,
                 filename: str,
                 fps: int = 10,
                 loop: bool = True
                 ) -> None:
        self.filename = filename
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

    def save(self) -> None:
        """
        Saves the captured frames as a GIF.

        Args:
            filename (str): filename to save GIF to
        """
        if self.loop:
            num_loops = 0
        else:
            num_loops = 1
        imageio.mimsave(self.filename, self.frames, fps=self.fps, loop=num_loops)
