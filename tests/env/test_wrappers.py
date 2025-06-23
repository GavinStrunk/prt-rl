import numpy as np
import pytest
import torch
import vmas
from prt_rl.env import wrappers
from prt_rl.env.interface import MultiAgentEnvParams, MultiGroupEnvParams
from prt_sim.jhu.bandits import KArmBandits
from prt_sim.jhu.robot_game import RobotGame

def test_jhu_wrapper_for_bandits():
    env = wrappers.JhuWrapper(environment=KArmBandits())

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

    action = np.array([[0]])
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (1, 1)
    assert reward.shape == (1, 1)
    assert done.shape == (1, 1)

def test_jhu_wrapper_for_robot_game():
    env = wrappers.JhuWrapper(environment=RobotGame(), render_mode="rgb_array")

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
    

    action = np.array([[0]])
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (1, 1)
    assert reward.shape == (1, 1)
    assert done.shape == (1, 1)
    assert info['rgb_array'].shape == (1, 800, 800, 3)
    assert info['rgb_array'].dtype == np.uint8

def test_gymnasium_wrapper_for_cliff_walking():
    # Reference: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = wrappers.GymnasiumWrapper(
        gym_name="CliffWalking-v0"
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

def test_vmas_wrapper():
    num_envs = 2
    env = wrappers.VmasWrapper(
        scenario="discovery",
        num_envs=num_envs,
    )

    assert isinstance(env.env, vmas.simulator.environment.environment.Environment)

    params = env.get_parameters()
    assert isinstance(params, MultiAgentEnvParams)
    assert params.num_agents == 5
    assert params.agent.action_len == 2
    assert params.agent.action_continuous is True
    assert params.agent.action_min == [-1.0, -1.0]
    assert params.agent.action_max == [1.0, 1.0]
    assert params.agent.observation_shape == (19,)
    assert params.agent.observation_continuous is True
    assert params.agent.observation_min == [-np.inf]*19
    assert params.agent.observation_max == [np.inf]*19

    state_td = env.reset()
    assert state_td.shape == (num_envs,)
    assert state_td['observation'].shape == (num_envs, params.num_agents, *params.agent.observation_shape)
    assert state_td['observation'].dtype == torch.float32

    action = state_td
    action['action'] = torch.zeros(num_envs, params.num_agents, params.agent.action_len)
    trajectory_td = env.step(action)
    assert trajectory_td.shape == (num_envs,)
    assert trajectory_td['next', 'reward'].shape == (num_envs, params.num_agents)
    assert trajectory_td['next', 'done'].shape == (num_envs, 1)

def test_multigroup_vmas_wrapper():
    num_envs = 1
    env = wrappers.VmasWrapper(
        scenario="kinematic_bicycle",
        num_envs=num_envs,
    )

    assert isinstance(env.env, vmas.simulator.environment.environment.Environment)

    params = env.get_parameters()
    assert isinstance(params, MultiGroupEnvParams)
    assert list(params.group.keys()) == ['bicycle', 'holo_rot']

    ma_bike = params.group['bicycle']
    assert ma_bike.num_agents == 1
    assert ma_bike.agent.action_len == 2
    assert ma_bike.agent.action_continuous is True
    assert ma_bike.agent.action_min == [-1.0, -0.5235987901687622]
    assert ma_bike.agent.action_max == [1.0, 0.5235987901687622]
    assert ma_bike.agent.observation_shape == (4,)
    assert ma_bike.agent.observation_continuous is True
    assert ma_bike.agent.observation_min == [-np.inf]*4
    assert ma_bike.agent.observation_max == [np.inf]*4

    ma_holo = params.group['holo_rot']
    assert ma_holo.num_agents == 1
    assert ma_holo.agent.action_len == 3
    assert ma_holo.agent.action_continuous is True
    assert ma_holo.agent.action_min == [-1.0, -1.0, -1.0]
    assert ma_holo.agent.action_max == [1.0, 1.0, 1.0]
    assert ma_holo.agent.observation_shape == (4,)
    assert ma_holo.agent.observation_continuous is True
    assert ma_holo.agent.observation_min == [-np.inf]*4
    assert ma_holo.agent.observation_max == [np.inf]*4
