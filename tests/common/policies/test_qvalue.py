import pytest
import torch.nn as nn
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies.qvalue import QValuePolicy
from prt_rl.common.networks import MLP, NatureCNNEncoder
from prt_rl.common.decision_functions import EpsilonGreedy, Softmax


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