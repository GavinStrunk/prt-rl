import numpy as np
import torch
import pytest
from typing import Dict
from prt_rl.common.buffers import ReplayBuffer, BaseBuffer, SumTree, PrioritizedReplayBuffer, RolloutBuffer


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
@pytest.fixture
def transition_batch():
    return {
        "state": torch.randn(10, 4),
        "action": torch.randint(0, 2, (10, 1)),
        "reward": torch.randn(10, 1),
        "done": torch.randint(0, 2, (10, 1), dtype=torch.bool),
        "next_state": torch.randn(10, 4),
    }

def test_init():
    buffer = ReplayBuffer(capacity=100, device=torch.device("cpu"))
    assert isinstance(buffer, BaseBuffer)
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

def test_resize_increases_capacity_and_preserves_data(transition_batch):
    buffer = ReplayBuffer(capacity=10)
    buffer.add(transition_batch)
    assert buffer.get_size() == 10
    old_data = {k: v.clone() for k, v in buffer.buffer.items()}
    
    # Resize to larger capacity
    buffer.resize(new_capacity=50)
    assert buffer.capacity == 50
    assert buffer.get_size() == 10
    assert buffer.pos == 10

    # Check that all old data was preserved
    for k in old_data:
        assert torch.allclose(buffer.buffer[k][:10], old_data[k])


def test_resize_error_on_shrink(transition_batch):
    buffer = ReplayBuffer(capacity=20)
    buffer.add(transition_batch)
    with pytest.raises(ValueError):
        buffer.resize(new_capacity=5)


def test_get_batches_returns_correct_batches(transition_batch):
    buffer = ReplayBuffer(capacity=50)
    buffer.add(transition_batch)
    buffer.add(transition_batch)  # 20 items total

    all_batches = list(buffer.get_batches(batch_size=7))
    assert sum(b["state"].shape[0] for b in all_batches) == 20
    assert all(set(b.keys()) == set(transition_batch.keys()) for b in all_batches)


def test_get_batches_empty():
    buffer = ReplayBuffer(capacity=10)
    batches = list(buffer.get_batches(batch_size=4))
    assert batches == []    

def test_sum_tree_add_and_total():
    tree = SumTree(capacity=4)
    tree.add(1.0)
    tree.add(2.0)
    tree.add(3.0)
    tree.add(4.0)
    assert tree.total_priority() == pytest.approx(10.0)

def test_sum_tree_wraparound():
    tree = SumTree(capacity=2)
    tree.add(1.0)
    tree.add(2.0)
    # Now wrap around and overwrite
    tree.add(3.0)
    assert tree.total_priority() == pytest.approx(5.0)

def test_sum_tree_update():
    tree = SumTree(capacity=4)
    tree.add(1.0)
    tree.add(2.0)
    tree.add(3.0)
    tree.add(4.0)
    idx = 2 + tree.capacity - 1  # third inserted item
    tree.update(idx, 10.0)
    assert tree.total_priority() == pytest.approx(10.0 + 1.0 + 2.0 + 4.0)

def test_sum_tree_get_leaf():
    tree = SumTree(capacity=4)
    tree.add(1.0)
    tree.add(1.0)
    tree.add(1.0)
    tree.add(7.0)
    # 70% of the mass is in the last element
    hits = [tree.get_leaf(np.random.uniform(0, tree.total_priority()))[2] for _ in range(1000)]
    # Count how often index 3 is hit
    assert hits.count(3) > 600  # At least 60% hits due to skewed distribution

def test_sum_tree_deterministic_get_leaf():
    tree = SumTree(capacity=4)
    tree.add(1.0)  # index 0
    tree.add(2.0)  # index 1
    tree.add(3.0)  # index 2
    tree.add(4.0)  # index 3
    total = tree.total_priority()
    leaf_idx, priority, data_idx = tree.get_leaf(0.1 * total)
    assert data_idx in [0, 1]
    leaf_idx, priority, data_idx = tree.get_leaf(0.9 * total)
    assert data_idx in [2, 3]


@pytest.fixture
def simple_transition():
    return {
        'state': torch.randn(1, 4),
        'action': torch.randint(0, 2, (1, 1)),
        'reward': torch.randn(1, 1),
        'next_state': torch.randn(1, 4),
        'done': torch.randint(0, 2, (1, 1), dtype=torch.bool)
    }

def test_add_and_size(simple_transition):
    buffer = PrioritizedReplayBuffer(capacity=10)
    buffer.add(simple_transition)
    assert buffer.get_size() == 1

def test_sample_structure(simple_transition):
    buffer = PrioritizedReplayBuffer(capacity=10)
    for _ in range(10):
        buffer.add(simple_transition)
    batch = buffer.sample(batch_size=4)
    assert set(batch.keys()) >= {'state', 'action', 'reward', 'next_state', 'done', 'weights', 'indices'}
    assert batch['state'].shape[0] == 4
    assert batch['weights'].shape == (4,)

def test_update_priorities(simple_transition):
    buffer = PrioritizedReplayBuffer(capacity=10)
    for _ in range(10):
        buffer.add(simple_transition)

    torch.manual_seed(0)
    np.random.seed(0)

    batch = buffer.sample(batch_size=4)
    old_weights = batch['weights'].clone()
    td_errors = torch.randn(4)
    buffer.update_priorities(batch['indices'], td_errors)
    batch2 = buffer.sample(batch_size=4)
    print("Old weights:", old_weights)
    print("New weights:", batch2['weights'])
    assert torch.allclose(torch.tensor([0.9014, 1.0000, 0.8295, 1.0000]), batch2['weights'], atol=1e-4)

def test_clear(simple_transition):
    buffer = PrioritizedReplayBuffer(capacity=10)
    buffer.add(simple_transition)
    buffer.clear()
    assert buffer.get_size() == 0
    assert buffer.buffer == {}


@pytest.fixture
def sample_experience() -> Dict[str, torch.Tensor]:
    return {
        "state": torch.randn(10, 4),
        "action": torch.randint(0, 2, (10, 1)),
        "reward": torch.randn(10, 1),
        "next_state": torch.randn(10, 4),
        "done": torch.zeros((10, 1), dtype=torch.bool)
    }

def test_init_and_add(sample_experience):
    buffer = RolloutBuffer(capacity=100)
    assert buffer.get_size() == 0
    buffer.add(sample_experience)
    assert buffer.get_size() == 10

def test_add_overflow(sample_experience):
    buffer = RolloutBuffer(capacity=15)
    buffer.add(sample_experience)  # 10 samples
    with pytest.raises(ValueError):
        buffer.add(sample_experience)  # would exceed 15

def test_sample_removal(sample_experience):
    buffer = RolloutBuffer(capacity=100)
    buffer.add(sample_experience)
    assert buffer.get_size() == 10

    batch = buffer.sample(batch_size=4)
    assert all(v.shape[0] == 4 for v in batch.values())
    assert buffer.get_size() == 6  # 10 - 4 removed

def test_sample_too_much(sample_experience):
    buffer = RolloutBuffer(capacity=10)
    buffer.add(sample_experience)
    with pytest.raises(ValueError):
        buffer.sample(batch_size=20)

def test_get_batches_exact_division():
    buffer = RolloutBuffer(capacity=30)
    for _ in range(3):
        buffer.add({
            "state": torch.randn(10, 4),
            "action": torch.randint(0, 2, (10, 1)),
            "reward": torch.randn(10, 1),
            "done": torch.zeros((10, 1), dtype=torch.bool)
        })

    batches = list(buffer.get_batches(batch_size=10))
    assert len(batches) == 3
    for batch in batches:
        assert all(v.shape[0] == 10 for v in batch.values())

def test_get_batches_inexact_division():
    buffer = RolloutBuffer(capacity=25)
    buffer.add({
        "state": torch.randn(23, 4),
        "action": torch.randint(0, 2, (23, 1)),
        "reward": torch.randn(23, 1),
        "done": torch.zeros((23, 1), dtype=torch.bool)
    })

    batches = list(buffer.get_batches(batch_size=8))
    assert len(batches) == 3
    sizes = [b["state"].shape[0] for b in batches]
    assert sizes == [8, 8, 7]

def test_get_batches_empty():
    buffer = RolloutBuffer(capacity=10)
    batches = list(buffer.get_batches(batch_size=4))
    assert batches == []

def test_clear_behavior():
    buffer = RolloutBuffer(capacity=50)
    buffer.add({
        "state": torch.randn(20, 4),
        "action": torch.randint(0, 2, (20,)),
        "reward": torch.randn(20),
        "done": torch.zeros(20, dtype=torch.bool)
    })
    buffer.clear()
    assert buffer.get_size() == 0
    assert not buffer.initialized