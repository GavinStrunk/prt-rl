import pytest
import torch
import torch.nn as nn
from prt_rl.common.policies.actor_critic import ActorCriticPolicy


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
    
class DummyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.called_predict = False
        self.called_eval = False

    def predict(self, state, deterministic=False):
        self.called_predict = True
        B = state.size(0)
        return torch.ones(B, 2), None, torch.zeros(B, 1)

    def evaluate_actions(self, state, action):
        self.called_eval = True
        B = state.size(0)
        return torch.zeros(B, 1), torch.ones(B, 1)

class DummyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)

    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def dummy_env_params():
    return DummyEnvParams()


@pytest.fixture
def dummy_state(dummy_env_params):
    return torch.randn(4, dummy_env_params.observation_shape[0])


def test_default_construction(dummy_env_params):
    policy = ActorCriticPolicy(env_params=dummy_env_params)
    assert isinstance(policy.actor, nn.Module)
    assert isinstance(policy.critic, nn.Module)

def test_custom_actor_and_critic(dummy_env_params):
    actor = DummyActor()
    critic = DummyCritic()
    policy = ActorCriticPolicy(env_params=dummy_env_params, actor=actor, critic=critic)
    assert policy.actor is actor
    assert policy.critic is critic

def test_shared_encoder(dummy_env_params, dummy_state):
    encoder = DummyEncoder(dummy_env_params.observation_shape)
    actor = DummyActor()
    critic = DummyCritic()
    policy = ActorCriticPolicy(env_params=dummy_env_params, encoder=encoder, actor=actor, critic=critic, share_encoder=True)
    # The encoder should NOT be deepcopied
    assert policy.critic_encoder is None

def test_unshared_encoder(dummy_env_params, dummy_state):
    encoder = DummyEncoder(dummy_env_params.observation_shape)
    actor = DummyActor()
    critic = DummyCritic()
    policy = ActorCriticPolicy(env_params=dummy_env_params, encoder=encoder, actor=actor, critic=critic, share_encoder=False)
    assert policy.critic_encoder is not None
    assert policy.critic_encoder is not encoder
    assert isinstance(policy.critic_encoder, DummyEncoder)

def test_predict_output_shape(dummy_env_params, dummy_state):
    actor = DummyActor()
    critic = DummyCritic()
    policy = ActorCriticPolicy(env_params=dummy_env_params, actor=actor, critic=critic)
    action, value, log_prob = policy.predict(dummy_state)
    assert action.shape == (4, 2)
    assert value.shape == (4, 1)
    assert log_prob.shape == (4, 1)

def test_forward_calls_predict(dummy_env_params, dummy_state):
    actor = DummyActor()
    critic = DummyCritic()
    policy = ActorCriticPolicy(env_params=dummy_env_params, actor=actor, critic=critic)
    action = policy.forward(dummy_state)
    assert actor.called_predict
    assert action.shape == (4, 2)

def test_evaluate_actions_outputs(dummy_env_params, dummy_state):
    actor = DummyActor()
    critic = DummyCritic()
    action = torch.randn(4, 2)
    policy = ActorCriticPolicy(env_params=dummy_env_params, actor=actor, critic=critic)
    value, log_probs, entropy = policy.evaluate_actions(dummy_state, action)
    assert actor.called_eval
    assert value.shape == (4, 1)
    assert log_probs.shape == (4, 1)
    assert entropy.shape == (4, 1)