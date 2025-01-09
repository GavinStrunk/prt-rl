import pytest
import torch
from tensordict.tensordict import TensorDict
from prt_rl.env.interface import EnvParams
from prt_rl.utils.policies import RandomPolicy, KeyboardPolicy, QTablePolicy, load_from_mlflow, GameControllerPolicy
from prt_rl.utils.qtable import QTable
from prt_rl.utils.decision_functions import Greedy


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
    params = EnvParams(
        action_shape=(1,),
        action_continuous=True,
        action_min=[1.0],
        action_max=[1.1],
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


def test_random_multiple_continuous_action_selection():
    params = EnvParams(
        action_shape=(3,),
        action_continuous=True,
        action_min=[0.0, 0.0, 0.0],
        action_max=[1.0, 1.0, 1.0],
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

    assert td['action'].shape == (1, 3)
    assert torch.allclose(td['action'], torch.tensor([[0.50, 0.77, 0.09]]), atol=1e-2)


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


# @pytest.mark.skip(reason="Requires a game controller input")
def test_game_controller_blocking_policy_with_discrete_actions():
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

    policy = GameControllerPolicy(
        env_params=params,
        blocking=True,
        key_action_map={
            GameControllerPolicy.Key.BUTTON_DPAD_UP: 0,
            GameControllerPolicy.Key.BUTTON_DPAD_DOWN: 1,
            GameControllerPolicy.Key.BUTTON_DPAD_RIGHT: 2,
            GameControllerPolicy.Key.BUTTON_DPAD_LEFT: 3,
            GameControllerPolicy.Key.BUTTON_X: 4,
        }
    )

    # Create fake observation TensorDict
    td = TensorDict({}, batch_size=[1])

    # Press dpad up to pass
    out_td = policy.get_action(td)
    assert out_td['action'][0] == 0

    # Press dpad down to pass
    out_td = policy.get_action(td)
    assert out_td['action'][0] == 1

    # Press dpad right to pass
    out_td = policy.get_action(td)
    assert out_td['action'][0] == 2

    # Press dpad left to pass
    out_td = policy.get_action(td)
    assert out_td['action'][0] == 3


@pytest.mark.skip(reason="Requires a game controller input")
def test_game_controller_nonblocking_policy_with_continuous_actions():
    params = EnvParams(
        action_shape=(2,),
        action_continuous=True,
        action_min=-2,
        action_max=2,
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    policy = GameControllerPolicy(
        env_params=params,
        key_action_map={
            'JOY_RIGHT_X': 0,
            'JOY_RIGHT_Y': 1,
        },
        blocking=False,
    )

    # Create fake observation TensorDict
    td = TensorDict({}, batch_size=[1])

    # You have to press up for this to pass
    action = 0
    while action == 0:
        td = policy.get_action(td)
        action = td['action'][0][0]
        print(f"action: {action}")
    assert td['action'][0][0] > 0


def test_qtable_policy():
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


@pytest.mark.skip(reason="Requires MLFlow server")
def test_mlflow_model_load():
    tracking_uri = 'http://localhost:5000'
    model_name = 'Robot Game'
    version = '2'

    policy = load_from_mlflow(tracking_uri=tracking_uri, model_name=model_name, model_version=version)
