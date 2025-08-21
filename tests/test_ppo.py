import torch
import pytest
from prt_rl.ppo import PPO
from prt_rl.env.wrappers import GymnasiumWrapper


def test_ppo_discrete_actions():
    env = GymnasiumWrapper("CartPole-v1")

    agent = PPO(
        env_params=env.get_parameters(),
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True  # If no exceptions are raised, the test passes

def test_ppo_continuous_actions():
    env = GymnasiumWrapper("InvertedPendulum-v5")

    agent = PPO(
        env_params=env.get_parameters(),
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True  # If no exceptions are raised, the test passes    

def test_ppo_multiple_envs():
    env = GymnasiumWrapper("CartPole-v1", num_envs=2)

    agent = PPO(
        env_params=env.get_parameters(),
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True    