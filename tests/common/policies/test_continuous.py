import pytest
import torch
import torch.nn as nn
from prt_rl.common.policies.continuous import ContinuousPolicy


class DummyEnvParams:
    def __init__(self,
                 observation_shape=(8,),
                 action_len=2,
                 action_min=-1.0,
                 action_max=1.0,
                 action_continuous=True):
        self.observation_shape = observation_shape
        self.action_len = action_len
        self.action_min = action_min
        self.action_max = action_max
        self.action_continuous = action_continuous


class DummyEncoder(nn.Module):
    def __init__(self, input_shape, features_dim=16):
        super().__init__()
        self.features_dim = features_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0], features_dim)
        )

    def forward(self, x):
        return self.net(x)

@pytest.fixture
def dummy_env_params():
    return DummyEnvParams()


@pytest.fixture
def dummy_state(dummy_env_params):
    return torch.randn(4, dummy_env_params.observation_shape[0])


class DummyPolicyHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    

def test_raises_on_discrete_action_space():
    env_params = DummyEnvParams((8,), 4, -1.0, 1.0, action_continuous=False)
    with pytest.raises(ValueError):
        _ = ContinuousPolicy(env_params)

def test_forward_continuous_policy_without_encoder(dummy_env_params, dummy_state):
    policy = ContinuousPolicy(
        env_params=dummy_env_params,
        encoder_network=None,
        policy_head=DummyPolicyHead
    )
    action = policy(dummy_state)
    assert action.shape == (4, dummy_env_params.action_len)
    assert torch.all(action <= dummy_env_params.action_max)
    assert torch.all(action >= dummy_env_params.action_min)

def test_forward_continuous_policy_with_encoder(dummy_env_params, dummy_state):
    policy = ContinuousPolicy(
        env_params=dummy_env_params,
        encoder_network=DummyEncoder,
        encoder_network_kwargs={"features_dim": 16},
        policy_head=DummyPolicyHead
    )
    action = policy(dummy_state)
    assert action.shape == (4, dummy_env_params.action_len)
    assert torch.all(action <= dummy_env_params.action_max)
    assert torch.all(action >= dummy_env_params.action_min)

def test_policy_head_respects_encoder_output_dim():
    # This test verifies that the policy head is constructed with the encoder's output dimension
    dummy_params = DummyEnvParams((5,), 3, -1, 1, True)
    encoder = DummyEncoder(input_shape=(5,), features_dim=7)
    policy = ContinuousPolicy(
        env_params=dummy_params,
        encoder_network=DummyEncoder,
        encoder_network_kwargs={"features_dim": 7},
        policy_head=DummyPolicyHead
    )
    assert policy.policy_head.linear.in_features == 7
    assert policy.policy_head.linear.out_features == 3