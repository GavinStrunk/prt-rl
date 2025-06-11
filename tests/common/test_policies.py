import pytest
import torch.nn as nn
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies import QValuePolicy
from prt_rl.common.networks import MLP, NatureCNNEncoder
from prt_rl.common.decision_functions import EpsilonGreedy, Softmax

def test_default_qvalue_policy_discrete_construction():
    # Discrete observation, discrete action
    params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=2,
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    # Initialize the QValuePolicy
    policy = QValuePolicy(env_params=params)
    assert policy.encoder_network == None
    assert isinstance(policy.policy_head, MLP)
    assert len(policy.policy_head.layers) == 5
    assert policy.policy_head.layers[0].in_features == 1
    assert policy.policy_head.layers[0].out_features == 64
    assert isinstance(policy.policy_head.layers[1], nn.ReLU)
    assert policy.policy_head.layers[2].in_features == 64
    assert policy.policy_head.layers[2].out_features == 64
    assert isinstance(policy.policy_head.layers[3], nn.ReLU)
    assert policy.policy_head.layers[4].in_features == 64
    assert policy.policy_head.layers[4].out_features == 3 
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

def test_qvalue_does_not_support_continuous_action():
    # Continuous action, discrete observation
    params = EnvParams(
        action_len=1,
        action_continuous=True,
        action_min=0,
        action_max=3,
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    # Initialize the QValuePolicy
    with pytest.raises(ValueError):
        QValuePolicy(env_params=params)

def test_qvalue_policy_with_policy():
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
    
    policy = QValuePolicy(
        env_params=params,
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
            "hidden_activation": None,
            "final_activation": None,
        }
    )
    assert isinstance(policy.encoder_network, NatureCNNEncoder)

    dummy_input = torch.rand((1, 4, 84, 84))
    action = policy(dummy_input)
    assert action.shape == (1, 1)  # Action shape should match the action_len of 1

def test_qvalue_policy_with_custom_decision_function():
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
    
    policy = QValuePolicy(
        env_params=params,
        decision_function=Softmax,
        decision_function_kwargs={
            "tau": 0.5,
        }
    )
    assert isinstance(policy.decision_function, Softmax)


# def test_qtable_policy():
#     # Create a fake environment that has 1 discrete action [0,...,3] and 1 discrete state with the same interval
#     params = EnvParams(
#         action_len=1,
#         action_continuous=False,
#         action_min=0,
#         action_max=3,
#         observation_shape=(1,),
#         observation_continuous=False,
#         observation_min=0,
#         observation_max=3,
#     )

#     policy = QTablePolicy(env_params=params)

#     # Check QTable is initialized properly
#     qt = policy.get_qtable()
#     assert qt.q_table.shape == (1, 4, 4)

#     # Check updating parameters
#     policy.set_parameter(name="epsilon", value=0.3)
#     assert policy.decision_function.epsilon == 0.3

#     # Check getting an action given an observation
#     obs_td = TensorDict({
#         "observation": torch.tensor([[1.0]], dtype=torch.int),
#     }, batch_size=[1])
#     action_td = policy.get_action(obs_td)
#     assert action_td['action'].shape == (1, 1)
#     assert action_td['action'].dtype == torch.int