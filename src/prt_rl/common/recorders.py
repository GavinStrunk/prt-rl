from abc import ABC
import imageio
import numpy as np
from prt_rl.common.buffers import ReplayBuffer


class Recorder(ABC):
    def reset(self) -> None:
        pass

    def record_info(self, info: dict) -> None:
        """
        Records information from the environment, such as rewards or other metrics.
        This method can be overridden by subclasses if needed.
        """
        pass
    def record_experience(self, experience: dict) -> None:
        """
        Records experience data, such as state, action, reward, and next state.
        This method can be overridden by subclasses if needed.
        """
        pass
    def close(self) -> None:
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

    def record_info(self, info: dict) -> None:
        if 'rgb_array' in info:
            # Get the frame from the first environment if there is more than one
            rgb_frame = info['rgb_array'][0]
            self._capture_frame(rgb_frame)

    def _capture_frame(self,
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

    def close(self) -> None:
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

class ExperienceRecorder(Recorder):
    """
    Records experience data such as state, action, reward, and next state.
    This can be used for training or analysis later.
    """
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.buffer = ReplayBuffer(capacity=1000000)

    def reset(self) -> None:
        self.buffer.clear()

    def record_experience(self, experience: dict) -> None:
        self.buffer.add(experience=experience)

    def close(self) -> None:
        self.buffer.save(self.filename)