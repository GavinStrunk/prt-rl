from tensordict.tensordict import TensorDict
import torch
from typing import Optional
from prt_sim.jhu.base import BaseEnvironment
from prt_rl.env.interface import EnvironmentInterface, EnvParams

class JhuWrapper(EnvironmentInterface):
    """
    Wraps the JHU environments in the Environment interface.

    The JHU environments are games and puzzles that were used in the JHU 705.741 RL course.
    """
    def __init__(self,
                 environment: BaseEnvironment,
                 render_mode: Optional[str] = None,
                 ) -> None:
        super().__init__(render_mode)
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
            observation_max=max(self.env.get_number_of_states()-1, 0),
        )
        return params

    def reset(self) -> TensorDict:
        state = self.env.reset()
        state_td = TensorDict(
            {
                'observation': torch.tensor([[state]], dtype=torch.int),
            },
            batch_size=torch.Size([1])
        )
        return state_td

    def step(self, action: TensorDict) -> TensorDict:
        action_val = action['action'][0].item()
        state, reward, done = self.env.execute_action(action_val)
        action['next'] = {
            'observation': torch.tensor([[state]], dtype=torch.int),
            'reward': torch.tensor([[reward]], dtype=torch.float),
            'done': torch.tensor([[done]], dtype=torch.bool),
        }

        if self.render_mode is not None:
            self.env.render()

        return action
