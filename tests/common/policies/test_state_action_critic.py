import pytest
import torch
import torch.nn as nn
from prt_rl.common.policies.state_action_critic import StateActionCritic


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
    
class DummyCriticHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def dummy_env_params():
    return DummyEnvParams()


@pytest.fixture
def dummy_state(dummy_env_params):
    return torch.randn(4, dummy_env_params.observation_shape[0])


@pytest.fixture
def dummy_action(dummy_env_params):
    return torch.randn(4, dummy_env_params.action_len)
        
def test_single_critic_forward_no_encoder(dummy_env_params, dummy_state, dummy_action):
    critic = StateActionCritic(
        env_params=dummy_env_params,
        num_critics=1,
        encoder=None,
        critic_head=DummyCriticHead
    )
    q = critic(dummy_state, dummy_action)
    assert isinstance(q, torch.Tensor)
    assert q.shape == (4, 1)

def test_single_critic_forward_with_encoder(dummy_env_params, dummy_state, dummy_action):
    encoder = DummyEncoder(dummy_env_params.observation_shape, features_dim=10)
    critic = StateActionCritic(
        env_params=dummy_env_params,
        num_critics=1,
        encoder=encoder,
        critic_head=DummyCriticHead,
        critic_head_kwargs={}
    )
    q = critic(dummy_state, dummy_action)
    assert isinstance(q, torch.Tensor)
    assert q.shape == (4, 1)

def test_multi_critic_forward_returns_tuple(dummy_env_params, dummy_state, dummy_action):
    critic = StateActionCritic(
        env_params=dummy_env_params,
        num_critics=3,
        encoder=None,
        critic_head=DummyCriticHead
    )
    q_values = critic(dummy_state, dummy_action)
    assert isinstance(q_values, tuple)
    assert len(q_values) == 3
    for q in q_values:
        assert isinstance(q, torch.Tensor)
        assert q.shape == (4, 1)

def test_forward_indexed_returns_single_output(dummy_env_params, dummy_state, dummy_action):
    critic = StateActionCritic(
        env_params=dummy_env_params,
        num_critics=2,
        encoder=None,
        critic_head=DummyCriticHead
    )
    q0 = critic.forward_indexed(0, dummy_state, dummy_action)
    q1 = critic.forward_indexed(1, dummy_state, dummy_action)

    assert isinstance(q0, torch.Tensor)
    assert q0.shape == (4, 1)
    assert isinstance(q1, torch.Tensor)
    assert q1.shape == (4, 1)

def test_forward_indexed_out_of_bounds(dummy_env_params, dummy_state, dummy_action):
    critic = StateActionCritic(
        env_params=dummy_env_params,
        num_critics=2,
        encoder=None,
        critic_head=DummyCriticHead
    )
    with pytest.raises(ValueError):
        _ = critic.forward_indexed(2, dummy_state, dummy_action)

def test_encoder_is_shared_across_critics(dummy_env_params):
    encoder = DummyEncoder(dummy_env_params.observation_shape, features_dim=10)
    critic = StateActionCritic(
        env_params=dummy_env_params,
        num_critics=2,
        encoder=encoder,
        critic_head=DummyCriticHead,
        critic_head_kwargs={}
    )
    assert critic.encoder is encoder
    assert all(isinstance(c, DummyCriticHead) for c in critic.critics)