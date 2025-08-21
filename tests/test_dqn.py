import pytest
from prt_rl.dqn import DQN, DoubleDQN
from prt_rl.env.wrappers import GymnasiumWrapper


def test_dqn_fails_on_continuous_actions():
    env = GymnasiumWrapper("InvertedPendulum-v5")

    with pytest.raises(ValueError):
        DQN(env_params=env.get_parameters())

def test_dqn_discrete_actions():
    env = GymnasiumWrapper("CartPole-v1")

    agent = DQN(
        env_params=env.get_parameters(),
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True  # If no exceptions are raised, the test passes