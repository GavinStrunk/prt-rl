import numpy as np
import pytest
import torch
import vmas
from prt_rl.env.wrappers.vmas_envs import VmasWrapper, VmasMultiGroupWrapper
from prt_rl.env.interface import MultiAgentEnvParams, MultiGroupEnvParams

def test_vmas_wrapper():
    num_envs = 2
    env = VmasWrapper(
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

    state, info = env.reset()
    assert state.shape == (num_envs, params.num_agents, *params.agent.observation_shape)
    assert state.dtype == torch.float32

    action = torch.zeros(num_envs, params.num_agents, params.agent.action_len)
    state, reward, done, info = env.step(action)
    assert state.shape == (num_envs, params.num_agents, *params.agent.observation_shape)
    assert reward.shape == (num_envs, params.num_agents)
    assert done.shape == (num_envs, 1)

def test_multigroup_vmas_wrapper_error():
    num_envs = 1
    with pytest.raises(ValueError):
        env = VmasWrapper(
            scenario="kinematic_bicycle",
            num_envs=num_envs,
        )

def test_vmas_multigroup_wrapper_env_params():
    num_envs = 1
    env = VmasMultiGroupWrapper(
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

def test_vmas_multigroup_wrapper_reset_step():
    num_envs = 3
    env = VmasMultiGroupWrapper(
        scenario="kinematic_bicycle",
        num_envs=num_envs,
        dict_spaces=False
    )

    state, info = env.reset()
    assert state['bicycle'].shape == (num_envs, 1, 4)
    assert state['holo_rot'].shape == (num_envs, 1, 4)

    action = {
        'bicycle': torch.zeros(num_envs, 1, 2),
        'holo_rot': torch.zeros(num_envs, 1, 3),
    }
    next_state, reward, done, info = env.step(action)
    assert next_state['bicycle'].shape == (num_envs, 1, 4)
    assert next_state['holo_rot'].shape == (num_envs, 1, 4)
    assert reward['bicycle'].shape == (num_envs, 1)
    assert reward['holo_rot'].shape == (num_envs, 1)
    assert done.shape == (num_envs, 1)