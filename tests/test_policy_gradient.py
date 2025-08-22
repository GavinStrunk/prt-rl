import torch
import pytest
from prt_rl.policy_gradient import PolicyGradient, PolicyGradientConfig
from prt_rl.env.wrappers import GymnasiumWrapper


def test_policy_gradient_discrete_actions():
    env = GymnasiumWrapper("CartPole-v1")

    agent = PolicyGradient(
        env_params=env.get_parameters(),
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True  # If no exceptions are raised, the test passes

def test_policy_gradient_continuous_actions():
    env = GymnasiumWrapper("InvertedPendulum-v5")

    agent = PolicyGradient(
        env_params=env.get_parameters(),
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True  # If no exceptions are raised, the test passes

def test_policy_gradient_multiple_envs():
    env = GymnasiumWrapper("CartPole-v1", num_envs=4)

    agent = PolicyGradient(
        env_params=env.get_parameters(),
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True

def test_policy_gradient_gae():
    env = GymnasiumWrapper("CartPole-v1")

    config = PolicyGradientConfig(
        use_gae=True,
    )

    agent = PolicyGradient(
        env_params=env.get_parameters(),
        config=config
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True

def test_policy_gradient_reward_to_go():
    env = GymnasiumWrapper("CartPole-v1")

    config = PolicyGradientConfig(
        use_reward_to_go=True,
    )

    agent = PolicyGradient(
        env_params=env.get_parameters(),
        config=config
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True

def test_policy_gradient_baseline():
    env = GymnasiumWrapper("CartPole-v1")

    config = PolicyGradientConfig(
        use_baseline=True,
    )

    agent = PolicyGradient(
        env_params=env.get_parameters(),
        config=config
    )

    # Test agent completes a training step without errors
    agent.train(env=env, total_steps=1)
    assert True

def test_compute_loss_output_shape():
    advantages = torch.tensor([[1.0], [0.5], [-0.5]])
    log_probs = torch.tensor([[0.2], [0.1], [-0.1]])

    loss = PolicyGradient._compute_loss(
        advantages=advantages,
        log_probs=log_probs,
        normalize=True
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar tensor

    loss = PolicyGradient._compute_loss(
        advantages=advantages,
        log_probs=log_probs,
        normalize=False
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar tensor