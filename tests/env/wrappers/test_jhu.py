import numpy as np
import pytest
import torch
from prt_rl.env import wrappers

def test_jhu_wrapper_for_bandits():
    env = wrappers.JhuWrapper(jhu_name="prt-sim/KArmBandits-v0")

    # Check the EnvParams are filled out correctly
    params = env.get_parameters()
    assert params.action_len == 1
    assert params.action_continuous is False
    assert params.action_min == 0
    assert params.action_max == 9
    assert params.observation_shape == (1,)
    assert params.observation_continuous is False
    assert params.observation_min == 0
    assert params.observation_max == 0

    # Check interface
    state, info = env.reset(seed=0)
    assert state.shape == (1, *params.observation_shape)

    # Check info
    assert info['optimal_bandit'].shape == (1, 1)
    assert info['optimal_bandit'] == np.array([[3]])
    assert info['bandits'].shape == (1, 10)
    np.testing.assert_allclose(info['bandits'], np.array([[1.7641,  0.4002,  0.9787,  2.2409,  1.8676, -0.9773,  0.9501, -0.1514, -0.1032,  0.4106]], dtype=np.float64), atol=1e-4)

    action = torch.tensor([[0]])
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (1, 1)
    assert reward.shape == (1, 1)
    assert done.shape == (1, 1)

def test_jhu_wrapper_for_robot_game():
    env = wrappers.JhuWrapper(jhu_name="prt-sim/RobotGame-v0", render_mode="rgb_array")

    # Check the EnvParams are filled out correctly
    params = env.get_parameters()
    assert params.action_len == 1
    assert params.action_continuous is False
    assert params.action_min == 0
    assert params.action_max == 3
    assert params.observation_shape == (1,)
    assert params.observation_continuous is False
    assert params.observation_min == 0
    assert params.observation_max == 10

    # Check interface
    state, info = env.reset()
    assert state.shape == (1, *params.observation_shape)
    

    action = torch.tensor([[0]])
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (1, 1)
    assert reward.shape == (1, 1)
    assert done.shape == (1, 1)
    assert info['rgb_array'].shape == (1, 800, 800, 3)
    assert info['rgb_array'].dtype == np.uint8