import pytest
import torch
from unittest.mock import MagicMock
from prt_rl.env.wrappers import GymnasiumWrapper
from prt_rl.common.collectors import SequentialCollector, ParallelCollector
import prt_rl.common.collectors as collectors

@pytest.fixture
def mock_env():
    env = MagicMock()
    # Returns (state, info)
    env.reset.return_value = (torch.zeros(1, 4, dtype=torch.float32), {})

    # Returns (next_state, reward, done, info)
    env.step.return_value = (torch.zeros(1, 4, dtype=torch.float32), torch.tensor([[1.0]]), torch.tensor([[False]]), {})
    return env

# Random Action Helper
# =========================================================
def test_random_discrete_action():
    env = GymnasiumWrapper("CartPole-v1")

    state, _ = env.reset()
    action = collectors.random_action(env.get_parameters(), state)
    assert action.shape == (1, 1)  # Single action for a single environment
    assert action.dtype == torch.int64  # Discrete action should be int64

# Sequential Collector Tests
# =========================================================
def test_collecting_experience():
    env = GymnasiumWrapper("CartPole-v1")

    collector = SequentialCollector(env)
    exp = collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64))

    assert exp['action'].shape == (1, 1)
    assert exp['reward'].shape == (1, 1)
    assert exp['state'].shape == (1, 4)
    assert exp['next_state'].shape == (1, 4)
    assert exp['done'].shape == (1, 1)

def test_collecting_random_experience():
    env = GymnasiumWrapper("CartPole-v1")

    collector = SequentialCollector(env)
    exp = collector.collect_experience()

    assert exp['action'].shape == (1, 1)
    assert exp['reward'].shape == (1, 1)
    assert exp['state'].shape == (1, 4)
    assert exp['next_state'].shape == (1, 4)
    assert exp['done'].shape == (1, 1)

def test_collecting_experience_update():
    env = GymnasiumWrapper("CartPole-v1")

    collector = SequentialCollector(env)
    exp = collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64))
    exp2 = collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64))
    assert torch.equal(exp['next_state'], exp2['state'])

def test_collecting_multiple_steps(mock_env):

    collector = SequentialCollector(mock_env)
    experience = collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64), num_steps=3)
    assert experience['state'].shape == (3, 4)

def test_environment_initial_reset(mock_env):
    collector = SequentialCollector(mock_env)
    experience_list = collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64))  
    assert mock_env.reset.call_count == 1
    assert mock_env.step.call_count == 1

def test_environment_reset_on_done(mock_env):
    # Fake the environment to return done on the first step
    mock_env.step.return_value = (torch.zeros(1, 4, dtype=torch.float32), torch.tensor([[1.0]]), torch.tensor([[True]]), {})

    collector = SequentialCollector(mock_env)

    # Resets when a done occurs during collection
    exp = collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64), num_steps=2)
    assert mock_env.reset.call_count == 2  # One for initial reset and one for done
    assert exp['action'].shape == (2, 1)

    # Reset occurs on subsequent call
    exp = collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64))
    assert mock_env.reset.call_count == 3  # One more reset after the done

def test_reward_tracking(mock_env):
    collector = SequentialCollector(mock_env)
    collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64), num_steps=5)

    assert collector.previous_episode_reward == 0
    assert collector.previous_episode_length == 0
    assert collector.current_episode_reward == 5
    assert collector.current_episode_length == 5
    assert collector.cumulative_reward == 5

    mock_env.step.return_value = (torch.zeros(1, 4, dtype=torch.float32), torch.tensor([[1.0]]), torch.tensor([[True]]), {})
    collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64), num_steps=1)
    mock_env.step.return_value = (torch.zeros(1, 4, dtype=torch.float32), torch.tensor([[1.0]]), torch.tensor([[False]]), {})
    collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64), num_steps=1)
    assert collector.previous_episode_reward == 6
    assert collector.previous_episode_length == 6
    assert collector.current_episode_reward == 1
    assert collector.current_episode_length == 1
    assert collector.cumulative_reward == 7

def test_sequential_trajectory_collection():
    env = GymnasiumWrapper("CartPole-v1")

    collector = SequentialCollector(env)
    trajectory, _ = collector.collect_trajectory(policy=lambda x: torch.zeros(1,1, dtype=torch.int64))

    assert len(trajectory) == 1
    assert trajectory[0]['done'][-1]

def test_sequential_multiple_trajectories():
    env = GymnasiumWrapper("CartPole-v1")

    collector = SequentialCollector(env)
    trajectories, _ = collector.collect_trajectory(policy=lambda x: torch.zeros(1,1, dtype=torch.int64), num_trajectories=3)

    assert len(trajectories) == 3
    for trajectory in trajectories:
        assert len(trajectory) > 0
        assert trajectory['done'][-1]

def test_sequential_trajectory_min_steps():
    env = GymnasiumWrapper("CartPole-v1")

    collector = SequentialCollector(env)
    trajectory, _ = collector.collect_trajectory(policy=lambda x: torch.zeros(1,1, dtype=torch.int64), min_num_steps=20)

    assert len(trajectory) > 1
    assert trajectory[-1]['done'][-1]  # Ensure the last step is done

# Parallel Collector Tests
# =========================================================
def test_parallel_collector():
    env = GymnasiumWrapper("CartPole-v1", num_envs=2)

    collector = ParallelCollector(env)
    exp = collector.collect_experience(policy=lambda x: torch.zeros(2,1, dtype=torch.int64), num_steps=10)

    assert exp['action'].shape == (10, 1)
    assert exp['reward'].shape == (10, 1)
    assert exp['state'].shape == (10, 4)
    assert exp['next_state'].shape == (10, 4)
    assert exp['done'].shape == (10, 1)

def test_parallel_collector_with_one_environment():
    env = GymnasiumWrapper("CartPole-v1", num_envs=1)

    collector = ParallelCollector(env)
    exp = collector.collect_experience(policy=lambda x: torch.zeros(1,1, dtype=torch.int64), num_steps=5)

    assert exp['action'].shape == (5, 1)
    assert exp['reward'].shape == (5, 1)
    assert exp['state'].shape == (5, 4)
    assert exp['next_state'].shape == (5, 4)
    assert exp['done'].shape == (5, 1)

def test_parallel_collector_without_flatten():
    env = GymnasiumWrapper("CartPole-v1", num_envs=2)

    collector = ParallelCollector(env, flatten=False)
    exp = collector.collect_experience(policy=lambda x: torch.zeros(2,1, dtype=torch.int64), num_steps=20)

    assert exp['action'].shape == (10, 2, 1)
    assert exp['reward'].shape == (10, 2, 1)
    assert exp['state'].shape == (10, 2, 4)
    assert exp['next_state'].shape == (10, 2, 4)
    assert exp['done'].shape == (10, 2, 1)


