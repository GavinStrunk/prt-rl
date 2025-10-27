import pytest
import torch
import torch.nn as nn
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies.distribution import DistributionPolicy
from prt_rl.common.distributions import Categorical, Normal
from prt_rl.common.networks import MLP


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