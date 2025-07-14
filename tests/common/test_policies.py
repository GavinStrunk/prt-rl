import pytest
import torch
import torch.nn as nn
from prt_rl.env.interface import EnvParams
from prt_rl.common.policies import QValuePolicy, ActorCriticPolicy, DistributionPolicy
from prt_rl.common.networks import MLP, NatureCNNEncoder
from prt_rl.common.decision_functions import EpsilonGreedy, Softmax
from prt_rl.common.distributions import Categorical, Normal

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
    
    dfcn = Softmax(tau=0.5)
    policy = QValuePolicy(
        env_params=params,
        decision_function=dfcn
    )
    assert isinstance(policy.decision_function, Softmax)

def test_actor_critic_encoder_fails_with_valid_keys():
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
    
    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            encoder_network={},  # MLP is not a valid encoder for ActorCriticPolicy
        )

    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            encoder_network={'actor': NatureCNNEncoder},  # NatureCNNEncoder is not a valid encoder for ActorCriticPolicy
        )
    
    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            encoder_network={'critic': NatureCNNEncoder},  # NatureCNNEncoder is not a valid encoder for ActorCriticPolicy
        )

    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            encoder_network={'actor': NatureCNNEncoder, 'critic': NatureCNNEncoder}, 
            encoder_network_kwargs={'actor': {'features_dim': 512}}, 
        )

    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            encoder_network={'actor': NatureCNNEncoder, 'critic': NatureCNNEncoder}, 
            encoder_network_kwargs={'critic': {'features_dim': 512}}, 
        )        

def test_actor_critic_policy_with_shared_nature_encoder():
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
    
    policy = ActorCriticPolicy(
        env_params=params,
        encoder_network=NatureCNNEncoder,
        encoder_network_kwargs={
            "features_dim": 512,
        },
    )
    
    assert isinstance(policy.actor_encoder_network, NatureCNNEncoder)
    assert isinstance(policy.critic_encoder_network, NatureCNNEncoder)
    assert policy.critic_encoder_network == policy.actor_encoder_network
    assert policy.actor_latent_dim == policy.critic_latent_dim

def test_actor_critic_policy_with_separate_nature_encoders():
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
    
    policy = ActorCriticPolicy(
        env_params=params,
        encoder_network={
            'actor': NatureCNNEncoder,
            'critic': NatureCNNEncoder
        },
        encoder_network_kwargs={
            'actor': {'features_dim': 512},
            'critic': {'features_dim': 256}
        }
    )
    
    assert isinstance(policy.actor_encoder_network, NatureCNNEncoder)
    assert isinstance(policy.critic_encoder_network, NatureCNNEncoder)
    assert policy.critic_encoder_network != policy.actor_encoder_network
    assert policy.actor_latent_dim == 512
    assert policy.critic_latent_dim == 256

def test_actor_critic_head_invalid_inputs():
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

    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            actor_critic_head={},  # Invalid type
        )

    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            actor_critic_head={'actor': MLP},  # Missing critic head
        )

    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            actor_critic_head={'critic': MLP},  # Missing actor head
        )
    
    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            actor_critic_head={'actor': MLP, 'critic': MLP},
            actor_critic_head_kwargs={'actor': {'network_arch': [64, 64]}},
        )

    with pytest.raises(ValueError):
        ActorCriticPolicy(
            env_params=params,
            actor_critic_head={'actor': MLP, 'critic': MLP},
            actor_critic_head_kwargs={'critic': {'network_arch': [64, 64]}},
        )
      

def test_actor_critic_same_head_network():
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

    policy = ActorCriticPolicy(
        env_params=params,
        actor_critic_head=MLP,
        actor_critic_head_kwargs={
            "network_arch": [64, 64],
            "hidden_activation": nn.Tanh(),
        }
    )
    
    assert isinstance(policy.actor_head, MLP)
    assert isinstance(policy.critic_head, MLP)
    assert policy.actor_head.layers[0].in_features == 3
    assert policy.actor_head.layers[0].out_features == 64
    assert isinstance(policy.actor_head.layers[1], nn.Tanh)
    assert policy.actor_head.layers[2].in_features == 64
    assert policy.actor_head.layers[2].out_features == 64
    assert isinstance(policy.actor_head.layers[3], nn.Tanh)


    assert policy.critic_head.layers[0].in_features == 3
    assert policy.critic_head.layers[0].out_features == 64
    assert isinstance(policy.critic_head.layers[1], nn.Tanh)
    assert policy.critic_head.layers[2].in_features == 64
    assert policy.critic_head.layers[2].out_features == 64
    assert isinstance(policy.critic_head.layers[3], nn.Tanh)
    assert policy.critic_head.layers[4].in_features == 64
    assert policy.critic_head.layers[4].out_features == 1


def test_actor_critic_with_different_heads():
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

    policy = ActorCriticPolicy(
        env_params=params,
        actor_critic_head={
            'actor': MLP,
            'critic': MLP
        },
        actor_critic_head_kwargs={
            'actor': {
                "network_arch": [64, 64],
                "hidden_activation": nn.ReLU(),
            },
            'critic': {
                "network_arch": [128, 64],
                "hidden_activation": nn.ReLU(),
            }
        }
    )
    
    assert isinstance(policy.actor_head, MLP)
    assert isinstance(policy.critic_head, MLP)
    assert policy.actor_head.layers[0].in_features == 3
    assert policy.actor_head.layers[0].out_features == 64
    assert isinstance(policy.actor_head.layers[1], nn.ReLU)
    assert policy.actor_head.layers[2].in_features == 64
    assert policy.actor_head.layers[2].out_features == 64
    assert isinstance(policy.actor_head.layers[3], nn.ReLU)
    assert policy.actor_feature_dim == 64

    assert policy.critic_head.layers[0].in_features == 3
    assert policy.critic_head.layers[0].out_features == 128
    assert isinstance(policy.critic_head.layers[1], nn.ReLU)
    assert policy.critic_head.layers[2].in_features == 128
    assert policy.critic_head.layers[2].out_features == 64
    assert isinstance(policy.critic_head.layers[3], nn.ReLU)
    assert policy.critic_head.layers[4].in_features == 64
    assert policy.critic_head.layers[4].out_features == 1

def test_actor_critic_default_distributions():
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

    policy = ActorCriticPolicy(env_params=params)
    assert policy.distribution is Categorical

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
    policy = ActorCriticPolicy(env_params=params)
    assert policy.distribution is Normal

def test_distribution_policy_default():
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

    policy = DistributionPolicy(env_params=params)
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

def test_distribution_policy_logits_fail_with_continuous_actions():
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
    
    with pytest.raises(ValueError):
        policy.get_logits(torch.tensor([[0.0, 0.0, 0.0]]))  # Should raise an error for continuous actions

def test_distribution_policy_logits_with_discrete_actions():
    params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=2,
        observation_shape=(3,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )
    
    policy = DistributionPolicy(env_params=params)
    logits = policy.get_logits(torch.tensor([[0.0, 0.0, 0.0]]))
    
    assert logits.shape == (1, 3)  # Should return logits for 3 discrete actions
    assert torch.all(logits >= 0) and torch.all(logits <= 1)  # Logits should be in the range [0, 1] for Categorical distribution

def test_distribution_policy_predict_action_and_log_probs():
    params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=2,
        observation_shape=(3,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )
    
    policy = DistributionPolicy(env_params=params)
    state = torch.tensor([[0.0, 0.0, 0.0]])
    
    action, log_probs = policy.predict(state)
    
    assert action.shape == (1, 1)  # Action shape should match the action_len of 1
    assert log_probs.shape == (1, 1)  # Log probabilities for 3 discrete actions
    assert torch.all(log_probs >= -float('inf')) and torch.all(log_probs <= 0)  # Log probabilities should be valid

def test_distribution_policy_predict_action_and_log_probs_continuous():
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
    
    action, log_probs = policy.predict(state)
    
    assert action.shape == (1, 2)  # Action shape should match the action_len of 2
    assert log_probs.shape == (1, 1)  # Log probabilities for continuous actions
    assert torch.all(log_probs >= -float('inf')) and torch.all(log_probs <= 0)  # Log probabilities should be valid

def test_distribution_policy_forward():
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

    torch.manual_seed(0)  # For reproducibility
    action1 = policy(state)

    torch.manual_seed(0)  # Reset seed to ensure same action is generated
    action2 = policy.forward(state)

    assert torch.equal(action1, action2)  # Both methods should return the same action