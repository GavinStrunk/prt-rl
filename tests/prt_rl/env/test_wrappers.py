import torch
from prt_rl.env import wrappers
from prt_sim.jhu.bandits import KArmBandits

def test_wrapper_for_bandits():
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
