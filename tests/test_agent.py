import torch
from tensordict.tensordict import TensorDict
from prt_rl.env.interface import EnvParams, MultiAgentEnvParams
from prt_rl.agent import RandomAgent

def test_random_discrete_action_selection():
    # Create a fake environment that has 1 discrete action [0,...,3] and 1 discrete state with the same interval
    params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=3,
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    policy = RandomAgent(env_params=params)

    # Set seed for consistent unit tests
    torch.manual_seed(3)

    # Create fake observation TensorDict
    action = policy(torch.zeros((1, 1)))

    assert action.shape == (1, 1)
    assert action[0] == 2


def test_random_continuous_action_selection():
    params = EnvParams(
        action_len=1,
        action_continuous=True,
        action_min=[1.0],
        action_max=[1.1],
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    policy = RandomAgent(env_params=params)

    # Set seed for consistent unit tests
    torch.manual_seed(0)

    # Create fake observation TensorDict
    action = policy(torch.zeros((1, 1)))

    assert action.shape == (1, 1)
    assert torch.allclose(action, torch.tensor([[1.05]]), atol=1e-2)

def test_random_multiple_continuous_action_selection():
    params = EnvParams(
        action_len=3,
        action_continuous=True,
        action_min=[0.0, 0.0, 0.0],
        action_max=[1.0, 1.0, 1.0],
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )

    policy = RandomAgent(env_params=params)

    # Set seed for consistent unit tests
    torch.manual_seed(0)

    # Create fake observation TensorDict
    action = policy(torch.zeros(1, 1))

    assert action.shape == (1, 3)
    assert torch.allclose(action, torch.tensor([[0.50, 0.77, 0.09]]), atol=1e-2)

def test_multi_agent_action_selection():
    params = EnvParams(
        action_len=3,
        action_continuous=True,
        action_min=[0.0, 0.0, 0.0],
        action_max=[1.0, 1.0, 1.0],
        observation_shape=(1,),
        observation_continuous=False,
        observation_min=0,
        observation_max=3,
    )
    ma_params = MultiAgentEnvParams(
        num_agents=2,
        agent=params
    )
    policy = RandomAgent(env_params=ma_params)

    # Set seed for consistent unit tests
    torch.manual_seed(0)

    action = policy(torch.zeros(1, 1))

    assert action.shape == (1, 2, 3)
    assert torch.allclose(action, torch.tensor([[[0.4963, 0.7682, 0.0885], [0.1320, 0.3074, 0.6341]]]), atol=1e-2)