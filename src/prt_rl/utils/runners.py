from typing import Optional
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.utils.policy import Policy
from prt_rl.utils.recorders import GifRecorder, Recorder


class Runner:
    """
    A runner executes a policy in an environment. It simplifies the process of evaluating policies that have been trained.

    @ todo add ability to show the visualization
    Args:
        env (EnvironmentInterface): the environment to run the policy in
        policy (Policy): the policy to run
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 policy: Policy,
                 ) -> None:
        self.env = env
        self.policy = policy

    def run(self,
            gif_filename: Optional[str] = None,
            ):
        # Set up the recorder
        if gif_filename is not None:
            recorder = GifRecorder()
        else:
            recorder = Recorder()
        recorder.reset()

        # Reset the environment
        state_td = self.env.reset()
        done = False

        # Loop until the episode is done
        while not done:
            action = self.policy.get_action(state_td)
            state_td = self.env.step(action)
            done = state_td['next', 'done']

            # Update the MDP
            state_td = self.env.step_mdp(state_td)

            recorder.capture_frame(state_td['rgb_array'][0].numpy())

        # Save the recording if there was one
        if gif_filename is not None:
            recorder.save(gif_filename)
