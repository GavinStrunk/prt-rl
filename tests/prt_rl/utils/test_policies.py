import pytest
import torch
from tensordict.tensordict import TensorDict
from prt_rl.env.interface import EnvParams
from prt_rl.utils.policies import RandomPolicy, KeyboardPolicy

def test_random_discrete_action_selection():
    # Create a fake environment that has 1 discrete action [0,...,3] and 1 discrete state with the same interval
    params = EnvParams(
        action_shape=(1,),
        action_continuous=False,
        action_min=0,
        action_max=3,
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    policy = RandomPolicy(env_params=params)

    # Set seed for consistent unit tests
    torch.manual_seed(3)

    # Create fake observation TensorDict
    td = TensorDict({}, batch_size=[1])
    td = policy.get_action(td)

    assert td['action'].shape == (1, 1)
    assert td['action'][0] == 2

def test_random_continuous_action_selection():
    # Create a fake environment that has 1 discrete action [0,...,3] and 1 discrete state with the same interval
    params = EnvParams(
        action_shape=(1,),
        action_continuous=True,
        action_min=1.0,
        action_max=1.1,
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    policy = RandomPolicy(env_params=params)

    # Set seed for consistent unit tests
    torch.manual_seed(0)

    # Create fake observation TensorDict
    td = TensorDict({}, batch_size=[1])
    td = policy.get_action(td)

    assert td['action'].shape == (1, 1)
    assert torch.allclose(td['action'], torch.tensor([[1.05]]), atol=1e-2)

@pytest.mark.skip(reason="Requires a keyboard press")
def test_keyboard_blocking_policy():
    # Create a fake environment that has 1 discrete action [0,...,3] and 1 discrete state with the same interval
    params = EnvParams(
        action_shape=(1,),
        action_continuous=True,
        action_min=1.0,
        action_max=1.1,
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    policy = KeyboardPolicy(
        env_params=params,
        key_action_map={
            'down': 0,
            'up': 1,
        }
    )

    # Create fake observation TensorDict
    td = TensorDict({}, batch_size=[1])

    # You have to press up for this to pass
    td = policy.get_action(td)
    assert td['action'][0] == 1
