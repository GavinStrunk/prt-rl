import pytest
from tensordict.tensordict import TensorDict
from prt_rl.env.interface import EnvParams
from prt_rl.common.policy import KeyboardPolicy

@pytest.mark.skip(reason="Requires a keyboard press")
def test_keyboard_blocking_policy():
    # Create a fake environment that has 1 discrete action [0,1] and 1 discrete state [0,...,3]
    params = EnvParams(
        action_shape=(1,),
        action_continuous=False,
        action_min=0,
        action_max=1,
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
    print("Press up arrow key to pass")
    td = policy.get_action(td)
    assert td['action'][0] == 1


@pytest.mark.skip(reason="Requires a keyboard press")
def test_keyboard_nonblocking_policy():
    # Create a fake environment that has 1 discrete action [0,1] and 1 discrete state [0,...,3]
    params = EnvParams(
        action_shape=(1,),
        action_continuous=False,
        action_min=0,
        action_max=1,
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
        },
        blocking=False,
    )

    # Create fake observation TensorDict
    td = TensorDict({}, batch_size=[1])

    # You have to press up for this to pass
    action = 0
    while action == 0:
        td = policy.get_action(td)
        action = td['action']
        print(f"action: {action}")
    assert td['action'][0] == 1