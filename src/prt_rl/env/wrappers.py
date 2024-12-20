from tensordict.tensordict import TensorDict
import torch
from prt_sim.jhu.base import BaseEnvironment
from prt_rl.env.interface import EnvironmentInterface, EnvParams

class JhuWrapper(EnvironmentInterface):
    """
    Wraps the JHU environments in the Environment interface.

    The JHU environments are games and puzzles that were used in the JHU 705.741 RL course.
    """
    def __init__(self,
                 environment: BaseEnvironment
                 ) -> None:
        self.env = environment

    def get_parameters(self) -> EnvParams:
        params = EnvParams(
            action_shape=(1,),
            action_continuous=False,
            action_min=0,
            action_max=self.env.get_number_of_actions()-1,
            observation_shape=(1,),
            observation_continuous=False,
            observation_min=0,
            observation_max=self.env.get_number_of_states()-1,
        )
        return params

    def reset(self) -> TensorDict:
        state = self.env.reset()
        state_td = TensorDict(
            {
                'observation': torch.tensor([[state]])
            },
            batch_size=torch.Size([1])
        )
        return state_td

    def step(self, action: TensorDict) -> TensorDict:
        action_val = action['action']
        state, reward, done = self.env.execute_action(action_val)
        trajectory_td = TensorDict(action)
        trajectory_td['next'] = {
            'observation': state,
            'reward': reward,
            'done': done,
        }
        return trajectory_td
