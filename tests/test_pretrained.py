import pytest

RUN_TESTS = False    # Set to False to skip tests

if not RUN_TESTS:
    pytest.skip("Skipping tests as RUN_TESTS is set to False", allow_module_level=True)

import os
import torch
from prt_rl.pretrained import SB3Agent

def test_sb3_agent_not_downloaded():
    agent = SB3Agent(
        model_dir="tests/logs/",
        model_type="ppo",
        env_name="CartPole-v1",
        device="cpu"
    )
    assert agent.model_path == "tests/logs/ppo/CartPole-v1_1/CartPole-v1.zip"
    assert os.path.isfile("tests/logs/ppo/CartPole-v1_1/CartPole-v1.zip")

def test_sb3_ppo_cartpole():
    agent = SB3Agent(
        model_dir="tests/logs/",
        model_type="ppo",
        env_name="CartPole-v1",
        device="cpu"
    )
    
    # Test prediction
    state = torch.zeros((1, 4), device="cpu")  # CartPole state has 4 dimensions
    action = agent.predict(state)
    assert action.shape == (1, 1)

    state = torch.zeros((4, 4), device="cpu")  # CartPole state has 4 dimensions
    actions = agent.predict(state)
    assert actions.shape == (4, 1)

def test_sb3_ppo_cartpole_gpu():
    agent = SB3Agent(
        model_dir="tests/logs/",
        model_type="ppo",
        env_name="CartPole-v1",
        device="cuda"
    )
    
    # Test prediction
    state = torch.zeros((1, 4), device="cuda")  # CartPole state has 4 dimensions
    action = agent.predict(state)
    assert action.shape == (1, 1)
    assert action.device.type == 'cuda'

    state = torch.zeros((4, 4), device="cuda")  # CartPole state has 4 dimensions
    actions = agent.predict(state)
    assert actions.shape == (4, 1)
    assert actions.device.type == 'cuda'

def test_sb3_continuous_environment():
    agent = SB3Agent(
        model_dir="tests/logs/",
        model_type="ppo",
        env_name="Pendulum-v1",
        device="cpu"
    )
    
    # Test prediction
    state = torch.zeros((1, 3), device="cpu")  # Pendulum state has 3 dimensions
    action = agent.predict(state)
    assert action.shape == (1, 1)

    state = torch.zeros((4, 3), device="cpu")  # Pendulum state has 3 dimensions
    actions = agent.predict(state)
    assert actions.shape == (4, 1)