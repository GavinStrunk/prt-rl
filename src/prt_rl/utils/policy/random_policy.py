from tensordict.tensordict import TensorDict
import torch
from prt_rl.env.interface import EnvParams
from prt_rl.utils.policy.policies import Policy

class RandomPolicy(Policy):
    """
    Implements a policy that uniformly samples random actions.

    Args:
        env_params (EnvParams): environment parameters
    """
    def __init__(self,
                 env_params: EnvParams,
                 ) -> None:
        super(RandomPolicy, self).__init__(env_params=env_params)

    def get_action(self,
                   state: TensorDict
                   ) -> TensorDict:
        """
        Randomly samples an action from action space.

        Returns:
            TensorDict: Tensordict with the "action" key added
        """
        if not self.env_params.action_continuous:
            # Add 1 to the high value because randint samples between low and 1 less than the high: [low,high)
            action = torch.randint(low=self.env_params.action_min, high=self.env_params.action_max + 1,
                                   size=(*state.batch_size, *self.env_params.action_shape))
        else:
            action = torch.rand(size=(*state.batch_size, *self.env_params.action_shape))

            # Scale the random [0,1] actions to the action space [min,max]
            max_actions = torch.tensor(self.env_params.action_max).unsqueeze(0)
            min_actions = torch.tensor(self.env_params.action_min).unsqueeze(0)
            action = action * (max_actions - min_actions) + min_actions

        state['action'] = action
        return state