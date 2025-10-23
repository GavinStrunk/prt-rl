import pytest
import torch
import torch.nn as nn
from prt_rl.common.policies.value_critic import ValueCritic


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


def test_forward_without_encoder(dummy_env_params, dummy_state):
    critic = ValueCritic(
        env_params=dummy_env_params,
        encoder=None,
        critic_head=DummyCriticHead
    )
    out = critic(dummy_state)
    assert out.shape == (4, 1)

def test_forward_with_encoder(dummy_env_params, dummy_state):
    critic = ValueCritic(
        env_params=dummy_env_params,
        encoder=DummyEncoder((8,), features_dim=10),
        critic_head=DummyCriticHead,
        critic_head_kwargs={}  # override since encoder changes dim
    )
    out = critic(dummy_state)
    assert out.shape == (4, 1)

def test_critic_head_output_matches_dim():
    env_params = DummyEnvParams(observation_shape=(6,))
    critic = ValueCritic(
        env_params=env_params,
        encoder=None,
        critic_head=DummyCriticHead
    )
    assert critic.critic_head.linear.in_features == 6
    assert critic.critic_head.linear.out_features == 1

def test_critic_respects_encoder_output_dim(dummy_state):
    # Encoder with output dim 12 should feed into critic_head expecting 12
    encoder = DummyEncoder((8,), features_dim=12)
    env_params = DummyEnvParams(observation_shape=(8,))
    critic = ValueCritic(
        env_params=env_params,
        encoder=encoder,
        critic_head=DummyCriticHead,
        critic_head_kwargs={}
    )
    out = critic(dummy_state)
    assert out.shape == (4, 1)