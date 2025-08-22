import torch
import pytest
from prt_rl.td3 import TD3Policy, TD3, TD3Config
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies import ContinuousPolicy, StateActionCritic
from prt_rl.env.wrappers import GymnasiumWrapper

@pytest.fixture
def dummy_env_params():
    return EnvParams(
        observation_shape=(4,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
        action_len=2,
        action_continuous=True,
        action_min=-1.0,
        action_max=1.0,
    )

def test_td3_policy_initialization(dummy_env_params):
    policy = TD3Policy(env_params=dummy_env_params, num_critics=2, device='cpu')

    assert isinstance(policy.actor, ContinuousPolicy)
    assert isinstance(policy.actor_target, ContinuousPolicy)
    assert isinstance(policy.critic, StateActionCritic)
    assert isinstance(policy.critic_target, StateActionCritic)
    assert policy.num_critics == 2

def test_td3_policy_forward_shape(dummy_env_params):
    policy = TD3Policy(env_params=dummy_env_params, device='cpu')

    batch_size = 5
    state = torch.randn(batch_size, *dummy_env_params.observation_shape)

    action = policy(state)
    assert action.shape == (batch_size, dummy_env_params.action_len)

def test_td3_policy_predict_shapes(dummy_env_params):
    policy = TD3Policy(env_params=dummy_env_params, device='cpu')

    batch_size = 3
    state = torch.randn(batch_size, *dummy_env_params.observation_shape)

    action, value_estimates, log_prob = policy.predict(state)

    # Check action shape
    assert action.shape == (batch_size, dummy_env_params.action_len)

    # Check value estimate shape
    assert value_estimates.ndim == 3  # (B, C, 1)
    assert value_estimates.shape[0] == batch_size
    assert value_estimates.shape[1] == policy.num_critics
    assert value_estimates.shape[2] == 1

    # Check log_prob is None
    assert log_prob is None

def test_td3_policy_deterministic(dummy_env_params):
    policy = TD3Policy(env_params=dummy_env_params, device='cpu')
    state = torch.randn(1, *dummy_env_params.observation_shape)

    a1 = policy(state, deterministic=True)
    a2 = policy(state, deterministic=True)

    assert torch.allclose(a1, a2, atol=1e-5), "Deterministic actions should match"


def test_td3_fails_on_discrete_actions():
    env = GymnasiumWrapper("CartPole-v1")

    with pytest.raises(ValueError):
        TD3(env_params=env.get_parameters())

def test_policy_gradient_continuous_actions():
    env = GymnasiumWrapper("InvertedPendulum-v5")

    config = TD3Config(
        steps_per_batch=1,
        min_buffer_size=1,
        mini_batch_size=1,
        delay_freq=1,
    )
    agent = TD3(env_params=env.get_parameters(), 
                config=config,
                )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True  # If no exceptions are raised, the test passes