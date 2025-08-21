import pytest
import torch
import torch.nn as nn
from typing import Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies import QValuePolicy, ActorCriticPolicy, DistributionPolicy, ContinuousPolicy, ValueCritic, StateActionCritic
from prt_rl.common.networks import MLP, NatureCNNEncoder
from prt_rl.common.decision_functions import EpsilonGreedy, Softmax
from prt_rl.common.distributions import Categorical, Normal

# =========================
# Dummy Classes
# =========================
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


class DummyPolicyHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class DummyCriticHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


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

# =========================
# Fixtures
# =========================
@pytest.fixture
def dummy_env_params():
    return DummyEnvParams()


@pytest.fixture
def dummy_state(dummy_env_params):
    return torch.randn(4, dummy_env_params.observation_shape[0])


@pytest.fixture
def dummy_action(dummy_env_params):
    return torch.randn(4, dummy_env_params.action_len)


@pytest.fixture
def dummy_encoder(dummy_env_params):
    return DummyEncoder(input_shape=dummy_env_params.observation_shape)


@pytest.fixture
def dummy_actor():
    return DummyActor()


@pytest.fixture
def dummy_critic():
    return DummyCritic()

@pytest.fixture
def discrete_env_params():
    return EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=3,
        observation_shape=(3,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )


@pytest.fixture
def continuous_env_params():
    return EnvParams(
        action_len=2,
        action_continuous=True,
        action_min=[0, 0],
        action_max=[1, 1],
        observation_shape=(3,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )

# =========================
# QValuePolicy Tests
# =========================
def test_default_qvalue_policy_discrete_construction(discrete_env_params):
    # Discrete observation, discrete action    
    # Initialize the QValuePolicy
    policy = QValuePolicy(env_params=discrete_env_params)
    assert policy.encoder_network == None
    assert isinstance(policy.policy_head, MLP)
    assert len(policy.policy_head.layers) == 5
    assert policy.policy_head.layers[0].in_features == 3
    assert policy.policy_head.layers[0].out_features == 64
    assert isinstance(policy.policy_head.layers[1], nn.ReLU)
    assert policy.policy_head.layers[2].in_features == 64
    assert policy.policy_head.layers[2].out_features == 64
    assert isinstance(policy.policy_head.layers[3], nn.ReLU)
    assert policy.policy_head.layers[4].in_features == 64
    assert policy.policy_head.layers[4].out_features == 4 
    assert policy.policy_head.final_activation == None
    assert isinstance(policy.decision_function, EpsilonGreedy)

def test_default_qvalue_policy_continuous_construction():
    # Continuous observation, discrete action
    params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=3,
        observation_shape=(3,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )

    # Initialize the QValuePolicy
    policy = QValuePolicy(env_params=params)
    assert policy.encoder_network == None
    assert isinstance(policy.policy_head, MLP)
    assert len(policy.policy_head.layers) == 5
    assert policy.policy_head.layers[0].in_features == 3
    assert policy.policy_head.layers[0].out_features == 64
    assert isinstance(policy.policy_head.layers[1], nn.ReLU)
    assert policy.policy_head.layers[2].in_features == 64
    assert policy.policy_head.layers[2].out_features == 64
    assert isinstance(policy.policy_head.layers[3], nn.ReLU)
    assert policy.policy_head.layers[4].in_features == 64
    assert policy.policy_head.layers[4].out_features == 4 
    assert policy.policy_head.final_activation == None
    assert isinstance(policy.decision_function, EpsilonGreedy)

def test_qvalue_does_not_support_continuous_action(continuous_env_params):
    # Continuous action, discrete observation
    # Initialize the QValuePolicy
    with pytest.raises(ValueError):
        QValuePolicy(env_params=continuous_env_params)

def test_qvalue_policy_with_policy(discrete_env_params):
    # Discrete observation, discrete action   
    policy = QValuePolicy(
        env_params=discrete_env_params,
        policy_head=MLP,
        policy_head_kwargs={
            "network_arch": [256, 256],
            "hidden_activation": nn.ReLU(),
            "final_activation": nn.Softmax(dim=-1),
            }
        )
    assert policy.encoder_network == None
    assert len(policy.policy_head.layers) == 5
    assert policy.policy_head.layers[0].in_features == 3
    assert policy.policy_head.layers[0].out_features == 256
    assert isinstance(policy.policy_head.layers[1], nn.ReLU)
    assert policy.policy_head.layers[2].in_features == 256
    assert policy.policy_head.layers[2].out_features == 256
    assert isinstance(policy.policy_head.layers[3], nn.ReLU)
    assert policy.policy_head.layers[4].in_features == 256
    assert policy.policy_head.layers[4].out_features == 4 
    assert isinstance(policy.policy_head.final_activation, nn.Softmax)
    assert isinstance(policy.decision_function, EpsilonGreedy)

def test_qvalue_policy_with_nature_encoder():
    import torch
    import numpy as np
    params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=3,
        observation_shape=(4, 84, 84),
        observation_continuous=True,
        observation_min=np.zeros((4, 84, 84)),
        observation_max=np.ones((4, 84, 84)) * 255,
    )
    policy = QValuePolicy(
        env_params=params,
        encoder_network=NatureCNNEncoder,
        encoder_network_kwargs={
            "features_dim": 512,
        },
        policy_head=MLP,
        policy_head_kwargs={
            "network_arch": None,
            "final_activation": None,
        }
    )
    assert isinstance(policy.encoder_network, NatureCNNEncoder)

    dummy_input = torch.rand((1, 4, 84, 84))
    action = policy(dummy_input)
    assert action.shape == (1, 1)  # Action shape should match the action_len of 1

def test_qvalue_policy_with_custom_decision_function(discrete_env_params):
    dfcn = Softmax(tau=0.5)
    policy = QValuePolicy(
        env_params=discrete_env_params,
        decision_function=dfcn
    )
    assert isinstance(policy.decision_function, Softmax)

# =========================
# DistributionPolicy Tests
# =========================
def test_distribution_policy_default(discrete_env_params, continuous_env_params):
    # Discrete action space
    policy = DistributionPolicy(env_params=discrete_env_params)
    assert issubclass(policy.distribution, Categorical)
    assert policy.encoder_network is None
    assert isinstance(policy.policy_head, MLP)
    assert policy.policy_head.layers[0].in_features == 3
    assert policy.policy_head.layers[0].out_features == 64
    assert isinstance(policy.policy_head.layers[1], nn.ReLU)
    assert policy.policy_head.layers[2].in_features == 64
    assert policy.policy_head.layers[2].out_features == 64
    assert isinstance(policy.policy_head.layers[3], nn.ReLU)
    assert policy.distribution_layer[0].in_features == 64
    assert policy.distribution_layer[0].out_features == 4
    assert isinstance(policy.distribution_layer[1], nn.Softmax)

    # Continuous action space
    policy = DistributionPolicy(env_params=continuous_env_params)
    assert issubclass(policy.distribution, Normal)
    assert policy.encoder_network is None
    assert isinstance(policy.policy_head, MLP)
    assert policy.policy_head.layers[0].in_features == 3
    assert policy.policy_head.layers[0].out_features == 64
    assert isinstance(policy.policy_head.layers[1], nn.ReLU)
    assert policy.policy_head.layers[2].in_features == 64
    assert policy.policy_head.layers[2].out_features == 64
    assert isinstance(policy.policy_head.layers[3], nn.ReLU)
    assert policy.distribution_layer.in_features == 64
    assert policy.distribution_layer.out_features == 2

def test_distribution_policy_logits_fail_with_continuous_actions(continuous_env_params):
    policy = DistributionPolicy(env_params=continuous_env_params)
    
    with pytest.raises(ValueError):
        policy.get_logits(torch.tensor([[0.0, 0.0, 0.0]]))  # Should raise an error for continuous actions

def test_distribution_policy_logits_with_discrete_actions(discrete_env_params):
    policy = DistributionPolicy(env_params=discrete_env_params)
    logits = policy.get_logits(torch.tensor([[0.0, 0.0, 0.0]]))
    
    assert logits.shape == (1, 4)  # Should return logits for 3 discrete actions
    assert torch.all(logits >= 0) and torch.all(logits <= 1)  # Logits should be in the range [0, 1] for Categorical distribution

def test_distribution_policy_predict_action_and_log_probs(discrete_env_params):
    policy = DistributionPolicy(env_params=discrete_env_params)
    state = torch.tensor([[0.0, 0.0, 0.0]])
    
    action, _, log_probs = policy.predict(state)
    
    assert action.shape == (1, 1)  # Action shape should match the action_len of 1
    assert log_probs.shape == (1, 1)  # Log probabilities for 3 discrete actions
    assert torch.all(log_probs >= -float('inf')) and torch.all(log_probs <= 0)  # Log probabilities should be valid

def test_distribution_policy_predict_action_and_log_probs_continuous(continuous_env_params):
    policy = DistributionPolicy(env_params=continuous_env_params)
    state = torch.tensor([[0.0, 0.0, 0.0]])
    
    action, _, log_probs = policy.predict(state)
    
    assert action.shape == (1, 2)  # Action shape should match the action_len of 2
    assert log_probs.shape == (1, 1)  # Log probabilities for continuous actions
    assert torch.all(log_probs >= -float('inf')) and torch.all(log_probs <= 0)  # Log probabilities should be valid

def test_distribution_policy_forward(continuous_env_params):
    policy = DistributionPolicy(env_params=continuous_env_params)
    state = torch.tensor([[0.0, 0.0, 0.0]])

    torch.manual_seed(0)  # For reproducibility
    action1 = policy(state)

    torch.manual_seed(0)  # Reset seed to ensure same action is generated
    action2 = policy.forward(state)

    assert torch.equal(action1, action2)  # Both methods should return the same action

def test_distribution_policy_evaluating_actions(continuous_env_params):
    policy = DistributionPolicy(env_params=continuous_env_params)
    state = torch.tensor([[0.0, 0.0, 0.0]])
    action = torch.tensor([[0.5]])

    log_probs, entropy = policy.evaluate_actions(state, action)
    assert log_probs.shape == (1, 1)  
    assert entropy.shape == (1, 1)

def test_distribution_policy_evaluating_multiple_actions():
    params = EnvParams(
        action_len=2,
        action_continuous=True,
        action_min=[0, 0],
        action_max=[1, 1],
        observation_shape=(3,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )
    
    policy = DistributionPolicy(env_params=params)
    state = torch.tensor([[0.0, 0.0, 0.0]])
    action = torch.tensor([[0.5, 0.5]])

    log_probs, entropy = policy.evaluate_actions(state, action)
    assert log_probs.shape == (1, 1)  # Log probabilities shape
    assert entropy.shape == (1, 1)    

# =========================
# ContinuousPolicy Tests
# =========================
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

# =========================
# ValueCritic Tests
# =========================   
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

# =========================
# StateActionCritic Tests
# =========================
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

# =========================
# ActorCriticPolicy Tests
# ========================= 
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