import torch
from prt_rl.env import wrappers
from prt_sim.jhu.bandits import KArmBandits
from prt_sim.jhu.robot_game import RobotGame

def test_jhu_wrapper_for_bandits():
    env = wrappers.JhuWrapper(environment=KArmBandits())

    # Check the EnvParams are filled out correctly
    params = env.get_parameters()
    assert params.action_shape == (1,)
    assert params.action_continuous == False
    assert params.action_min == 0
    assert params.action_max == 9
    assert params.observation_shape == (1,)
    assert params.observation_continuous == False
    assert params.observation_min == 0
    assert params.observation_max == 0

    # Check interface
    state_td = env.reset()
    assert state_td.shape == (1,)
    assert state_td['observation'].shape == (1, *params.observation_shape)

    action = state_td
    action['action'] = torch.tensor([[0]])
    trajectory_td = env.step(action)
    print(trajectory_td)

def test_jhu_wrapper_for_robot_game():
    env = wrappers.JhuWrapper(environment=RobotGame())

    # Check the EnvParams are filled out correctly
    params = env.get_parameters()
    assert params.action_shape == (1,)
    assert params.action_continuous == False
    assert params.action_min == 0
    assert params.action_max == 3
    assert params.observation_shape == (1,)
    assert params.observation_continuous == False
    assert params.observation_min == 0
    assert params.observation_max == 10

    # Check interface
    state_td = env.reset()
    assert state_td.shape == (1,)
    assert state_td['observation'].shape == (1, *params.observation_shape)

    action = state_td
    action['action'] = torch.tensor([[0]])
    trajectory_td = env.step(action)
    print(trajectory_td)

def test_gymnasium_wrapper_for_cliff_walking():
    # Reference: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = wrappers.GymnasiumWrapper(
        gym_name="CliffWalking-v0"
    )

    params = env.get_parameters()
    assert params.action_shape == (1,)
    assert params.action_continuous == False
    assert params.action_min == 0
    assert params.action_max == 3
    assert params.observation_shape == (1,)
    assert params.observation_continuous == False
    assert params.observation_min == 0
    assert params.observation_max == 47

    state_td = env.reset()
    assert state_td.shape == (1,)
    assert state_td['observation'].shape == (1, *params.observation_shape)

    action = state_td
    action['action'] = torch.tensor([[0]])
    trajectory_td = env.step(action)
    print(trajectory_td)