from tensordict.tensordict import TensorDict
import torch
from prt_rl.env.interface import EnvParams
from prt_rl.common.policy import QTablePolicy

def test_qtable_policy():
    # Create a fake environment that has 1 discrete action [0,...,3] and 1 discrete state with the same interval
    params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=3,
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    policy = QTablePolicy(env_params=params)

    # Check QTable is initialized properly
    qt = policy.get_qtable()
    assert qt.q_table.shape == (1, 4, 4)

    # Check updating parameters
    policy.set_parameter(name="epsilon", value=0.3)
    assert policy.decision_function.epsilon == 0.3

    # Check getting an action given an observation
    obs_td = TensorDict({
        "observation": torch.tensor([[1.0]], dtype=torch.int),
    }, batch_size=[1])
    action_td = policy.get_action(obs_td)
    assert action_td['action'].shape == (1, 1)
    assert action_td['action'].dtype == torch.int