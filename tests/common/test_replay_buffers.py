import torch
import pytest
from typing import Dict
from prt_rl.common.replay_buffers import ReplayBuffer, BaseReplayBuffer


@pytest.fixture
def example_transition():
    torch.manual_seed(0)
    return {
        "state": torch.randn(4, 8),
        "action": torch.randn(4, 3),
        "reward": torch.randn(4, 1),
        "done": torch.zeros(4, 1),
        "next_state": torch.randn(4, 8),
    }


def test_init():
    buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))
    assert isinstance(buffer, BaseReplayBuffer)
    assert len(buffer) == 0


def test_add_and_sample(example_transition):
    buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))

    for _ in range(5):
        buffer.add(example_transition)

    assert len(buffer) == 20  # 5 batches of 4

    batch = buffer.sample(batch_size=10)
    assert isinstance(batch, dict)
    assert set(batch.keys()) == set(example_transition.keys())
    for k in batch:
        assert batch[k].shape[0] == 10


def test_capacity_limit():
    buffer = ReplayBuffer(capacity=16, device=torch.device("cpu"))

    # Insert more than capacity to test overwrite
    for _ in range(5):
        batch = {
            "state": torch.randn(4, 8),
            "action": torch.randn(4, 3),
            "reward": torch.randn(4, 1),
            "done": torch.zeros(4, 1),
            "next_state": torch.randn(4, 8),
        }
        buffer.add(batch)

    assert len(buffer) == 16

    # Should be no out-of-bounds or stale entries
    sample = buffer.sample(batch_size=8)
    for k in sample:
        assert sample[k].shape == (8,) + buffer.buffer[k].shape[1:]


def test_sample_too_early(example_transition):
    buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))

    buffer.add(example_transition)  # 4 samples
    with pytest.raises(ValueError):
        buffer.sample(batch_size=10)


def test_device_consistency(example_transition):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = ReplayBuffer(capacity=100, device=device)

    transition = {k: v.to(device) for k, v in example_transition.items()}
    buffer.add(transition)
    batch = buffer.sample(batch_size=2)

    for v in batch.values():
        assert v.device.type == torch.device(device).type

def test_replay_buffer_clear():
    buffer = ReplayBuffer(capacity=100)

    # Add dummy transition
    transition = {
        "state": torch.randn(5, 4),
        "action": torch.randint(0, 2, (5, 1)),
        "reward": torch.randn(5, 1),
        "next_state": torch.randn(5, 4),
        "done": torch.randint(0, 2, (5, 1), dtype=torch.bool),
    }
    buffer.add(transition)

    assert len(buffer) == 5
    assert buffer.initialized
    assert buffer.buffer  # buffer dict should be populated

    # Clear the buffer
    buffer.clear()

    assert len(buffer) == 0
    assert not buffer.initialized
    assert buffer.buffer == {}

    # After clearing, adding should reinitialize
    buffer.add(transition)
    assert len(buffer) == 5