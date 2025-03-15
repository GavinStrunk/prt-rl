from typing import Optional
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.policy import Policy
from prt_rl.common.recorders import Recorder
from prt_rl.common.visualizers import Visualizer


class Runner:
    """
    A runner executes a policy in an environment. It simplifies the process of evaluating policies that have been trained.

    Args:
        env (EnvironmentInterface): the environment to run the policy in
        policy (Policy): the policy to run
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 policy: Policy,
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
        state_td = self.env.reset()
        done = False

        # Start visualizer and show initial frame
        self.visualizer.start()
        rgb_frame = state_td['rgb_array'][0].numpy()
        self.recorder.capture_frame(rgb_frame)
        self.visualizer.show(rgb_frame)

        # Loop until the episode is done
        while not done:
            action = self.policy.get_action(state_td)
            state_td = self.env.step(action)
            done = state_td['next', 'done']

            # Update the MDP
            state_td = self.env.step_mdp(state_td)

            # Record the environment frame
            rgb_frame = state_td['rgb_array'][0].numpy()
            self.recorder.capture_frame(rgb_frame)
            self.visualizer.show(rgb_frame)

        self.visualizer.stop()
        # Save the recording
        self.recorder.save()
