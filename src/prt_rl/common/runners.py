from typing import Optional
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.policy import Policy
from prt_rl.common.recorders import Recorder
from prt_rl.common.visualizers import Visualizer


class Runner:
    """
    A runner executes a policy in an environment. It simplifies the process of evaluating policies that have been trained.

    The runner assumes the rgb_array is in the info dictioanary and has shape (num_envs, channel, height, width).
    Args:
        env (EnvironmentInterface): the environment to run the policy in
        policy (Policy): the policy to run
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 policy: object,
                 recorder: Optional[Recorder] = None,
                 visualizer: Optional[Visualizer] = None,
                 ) -> None:
        self.env = env
        self.policy = policy
        self.recorder = recorder or Recorder()
        self.visualizer = visualizer or Visualizer()

    def run(self):
        # Reset the environment and recorder
        self.recorder.reset()
        state, info = self.env.reset()
        done = False

        # Start visualizer and show initial frame
        self.visualizer.start()
        rgb_frame = info['rgb_array'][0]
        rgb_frame = rgb_frame.transpose(1, 2, 0)
        self.recorder.capture_frame(rgb_frame)
        self.visualizer.show(rgb_frame)

        # Loop until the episode is done
        while not done:
            action = self.policy.predict(state)
            next_state, reward, done, info = self.env.step(action)

            # Record the environment frame
            rgb_frame = info['rgb_array'][0]
            rgb_frame = rgb_frame.transpose(1, 2, 0)
            self.recorder.capture_frame(rgb_frame)
            self.visualizer.show(rgb_frame)

        self.visualizer.stop()
        # Save the recording
        self.recorder.save()
