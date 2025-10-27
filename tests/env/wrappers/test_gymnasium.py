import numpy as np
import pytest
import torch
from prt_rl.env import wrappers
import prt_sim.gymnasium 

def test_gymnasium_wrapper_for_cliff_walking():
    # Reference: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = wrappers.GymnasiumWrapper(
        gym_name="CliffWalking-v1"
    )

    params = env.get_parameters()
    assert params.action_len == 1
    assert params.action_continuous is False
    assert params.action_min == 0
    assert params.action_max == 3
    assert params.observation_shape == (1,)
    assert params.observation_continuous is False
    assert params.observation_min == 0
    assert params.observation_max == 47

    state, info = env.reset()
    assert state.shape == (1, *params.observation_shape)
    assert state.dtype == torch.int64

    action = torch.tensor([[0]])
    assert action.shape == (1, 1)
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (1, 1)
    assert reward.shape == (1, 1)
    assert done.shape == (1, 1)
    assert info == {'prob': 1.0}

def test_gymnasium_wrapper_continuous_observations():
    env = wrappers.GymnasiumWrapper(
        gym_name="MountainCar-v0",
        render_mode=None,
    )

    params = env.get_parameters()
    assert params.action_len == 1
    assert params.action_continuous is False
    assert params.action_min == 0
    assert params.action_max == 2
    assert params.observation_shape == (2,)
    assert params.observation_continuous is True
    assert len(params.observation_min) == 2
    assert params.observation_min[0] == pytest.approx(-1.2)
    assert params.observation_min[1] == pytest.approx(-0.07)
    assert len(params.observation_max) == 2
    assert params.observation_max[0] == pytest.approx(0.6)
    assert params.observation_max[1] == pytest.approx(0.07)

    state, info = env.reset()
    assert state.shape == (1, *params.observation_shape)
    assert state.dtype == torch.float32

    action = torch.zeros(1, params.action_len, dtype=torch.int)
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (1, 2)
    assert reward.shape == (1, 1)
    assert done.shape == (1, 1)

def test_gymnasium_wrapper_continuous_actions():
    env = wrappers.GymnasiumWrapper(
        gym_name="MountainCarContinuous-v0",
        render_mode=None,
    )

    params = env.get_parameters()
    assert params.action_len == 1
    assert params.action_continuous is True
    assert params.action_min == [-1]
    assert params.action_max == [1.0]
    assert params.observation_shape == (2,)
    assert params.observation_continuous is True
    assert params.observation_min == pytest.approx([-1.2, -0.07])
    assert params.observation_max == pytest.approx([0.6, 0.07])

    state, info = env.reset()
    assert state.shape == (1, *params.observation_shape)
    assert state.dtype == torch.float32

    action = torch.zeros(1, params.action_len)
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (1, 2)
    assert reward.shape == (1, 1)
    assert done.shape == (1, 1)

def test_gymnasium_multienv():
    num_envs = 5
    env = wrappers.GymnasiumWrapper(
        gym_name="MountainCarContinuous-v0",
        num_envs=num_envs,
        render_mode=None,
    )
    params = env.get_parameters()
    assert params.action_len == 1
    assert params.action_continuous is True
    assert params.action_min == [-1]
    assert params.action_max == [1.0]
    assert params.observation_shape == (2,)
    assert params.observation_continuous is True
    assert params.observation_min == pytest.approx([-1.2, -0.07])
    assert params.observation_max == pytest.approx([0.6, 0.07])

    state, info = env.reset()
    assert state.shape == (num_envs, *params.observation_shape)
    assert state.dtype == torch.float32

    action = torch.zeros(num_envs, params.action_len)
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (num_envs, *params.observation_shape)
    assert reward.shape == (num_envs, 1)
    assert done.shape == (num_envs, 1)

def test_gymnasium_discrete_multienv():
    num_envs = 4
    env = wrappers.GymnasiumWrapper(
        gym_name="CartPole-v1",
        num_envs=num_envs,
        render_mode=None,
    )

    params = env.get_parameters()
    assert params.action_len == 1
    assert params.action_continuous is False
    assert params.action_min == 0
    assert params.action_max == 1
    assert params.observation_shape == (4,)
    assert params.observation_continuous is True
    assert len(params.observation_min) == 4
    assert len(params.observation_max) == 4

    state, _ = env.reset()
    assert state.shape == (num_envs, *params.observation_shape)
    assert state.dtype == torch.float32

    action = torch.zeros(num_envs, params.action_len, dtype=torch.int)
    next_state, reward, done, _ = env.step(action)
    assert next_state.shape == (num_envs, *params.observation_shape)
    assert reward.shape == (num_envs, 1)
    assert done.shape == (num_envs, 1)

def test_gymnasium_reset_done():
    import copy
    env = wrappers.GymnasiumWrapper(
        gym_name="CartPole-v1",
        render_mode=None,
        num_envs=4
    )

    state, _ = env.reset(seed=0)
    assert state.shape == (4, 4)

    action = torch.zeros(4, 1, dtype=torch.int)
    next_state, reward, done, info = env.step(action)

    # Reset only the second and third environments
    new_state = copy.deepcopy(next_state)
    new_state[1], _ = env.reset_index(1, seed=1)
    new_state[2], _ = env.reset_index(2, seed=2)
    torch.testing.assert_close(new_state[1:3], state[1:3], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(new_state[0], state[0], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(new_state[3], state[3], rtol=1e-6, atol=1e-6)

def test_gymnasium_wrapper_with_render():
    env = wrappers.GymnasiumWrapper(
        gym_name="CartPole-v1",
        render_mode="rgb_array",
    )

    state, info = env.reset()
    assert info['rgb_array'].shape == (1, 400, 600, 3)

    action = torch.zeros((1, 1), dtype=torch.int)
    next_state, reward, done, info = env.step(action)
    assert info['rgb_array'].shape == (1, 400, 600, 3)

def test_gymnasium_mujoco_types():
    env = wrappers.GymnasiumWrapper(
        gym_name="InvertedPendulum-v5",
        render_mode=None,
    )

    state, info = env.reset()
    assert state.shape == (1, 4)
    assert state.dtype == torch.float32

def test_gymnasium_get_params_from_dict():
    from gymnasium import spaces

    action_space = spaces.Dict({
            "algorithm": spaces.Discrete(3),
            "parameters": spaces.Box(low=0.0, high=1.0, shape=(5,))  
        })  

    act_len, act_cont, act_min, act_max = wrappers.GymnasiumWrapper._get_params_from_dict(action_space, is_action=True)

    assert act_len == 6
    assert act_cont == [False, True, True, True, True, True]
    assert act_min == [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert act_max == [2, 1.0, 1.0, 1.0, 1.0, 1.0]

@pytest.mark.skip("ImagePipeline-v0 requires downloading the BDD100K dataset.")
def test_gymnasium_image_pipeline_dict():
    env = wrappers.GymnasiumWrapper(
        gym_name="prt-sim/ImagePipeline-v0"
    )
    params = env.get_parameters()
    assert params.action_len == 6
    assert params.action_continuous == [False, True, True, True, True, True]
    assert params.action_min == [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert params.action_max == [3, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert params.observation_shape == (3, 720, 1280)
    assert params.observation_continuous is True
    assert params.observation_min == np.zeros((3, 720, 1280)).tolist()
    assert params.observation_max == np.ones((3, 720, 1280)).tolist()

