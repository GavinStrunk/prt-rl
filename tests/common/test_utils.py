import torch
import torch.nn as nn
import pytest
from copy import deepcopy
import prt_rl.common.utils as utils

# Polyak update tests
# ==================================================
class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

def test_polyak_update_tau_zero():
    net = DummyNet()
    target = DummyNet()
    original = deepcopy(target)
    utils.polyak_update(target, net, tau=0.0)
    for p1, p2 in zip(target.parameters(), original.parameters()):
        assert torch.allclose(p1.data, p2.data), "tau=0 should leave target unchanged"

def test_polyak_update_tau_one():
    net = DummyNet()
    target = DummyNet()
    utils.polyak_update(target, net, tau=1.0)
    for p1, p2 in zip(target.parameters(), net.parameters()):
        assert torch.allclose(p1.data, p2.data), "tau=1 should copy exactly from network"

def test_polyak_update_in_place():
    net = DummyNet()
    target = DummyNet()
    target_copy = deepcopy(target)
    utils.polyak_update(target, net, tau=0.5)
    for p1, p2 in zip(target.parameters(), target_copy.parameters()):
        assert not torch.allclose(p1.data, p2.data), "Parameters should have changed"

def test_polyak_update_correctness():
    net = DummyNet()
    target = DummyNet()
    tau = 0.3

    # Manually compute expected update
    expected_params = [
        tau * p_net.data + (1 - tau) * p_target.data
        for p_net, p_target in zip(net.parameters(), target.parameters())
    ]

    utils.polyak_update(target, net, tau=tau)

    for actual, expected in zip(target.parameters(), expected_params):
        assert torch.allclose(actual.data, expected), "Incorrect Polyak update"

# Hard update tests
# ==================================================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 2)

def test_hard_update_copies_parameters_correctly():
    net = SimpleNet()
    target = SimpleNet()

    # Modify original target so it's different from net
    for param in target.parameters():
        param.data.fill_(0.1)

    utils.hard_update(target, net)

    for p1, p2 in zip(target.parameters(), net.parameters()):
        assert torch.allclose(p1.data, p2.data), "Parameters were not copied correctly"

def test_hard_update_is_in_place():
    net = SimpleNet()
    target = SimpleNet()
    target_copy = deepcopy(target)

    # Perform update
    utils.hard_update(target, net)

    # Check that target parameters have changed from original
    for updated, original in zip(target.parameters(), target_copy.parameters()):
        assert not torch.allclose(updated.data, original.data), "Update did not happen in-place"

def test_hard_update_uses_different_memory():
    net = SimpleNet()
    target = SimpleNet()
    utils.hard_update(target, net)

    for p1, p2 in zip(target.parameters(), net.parameters()):
        assert p1.data.data_ptr() != p2.data.data_ptr(), "Parameters share the same memory"

def test_hard_update_raises_on_mismatched_shapes():
    class MismatchedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 4)  # shape mismatch
            self.linear2 = nn.Linear(4, 2)

    net = MismatchedNet()
    target = SimpleNet()

    with pytest.raises(RuntimeError):  # torch will raise if shape mismatch during copy_
        utils.hard_update(target, net)

# GAE tests
# ==================================================
def test_gae_time_major():
    rewards = torch.tensor([[[1.0]], [[1.0]], [[1.0]]])
    values = torch.tensor([[[0.5]], [[0.5]], [[0.5]]])
    dones = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])
    last_values = torch.tensor([[0.0]])

    adv, ret = utils.generalized_advantage_estimates(rewards, values, dones, last_values)

    assert adv.shape == rewards.shape
    assert ret.shape == rewards.shape
    assert torch.allclose(ret, adv + values)

def test_gae_flattened():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    last_values = torch.tensor([0.0])

    adv, ret = utils.generalized_advantage_estimates(rewards, values, dones, last_values)

    assert adv.shape == rewards.shape
    assert ret.shape == rewards.shape
    assert torch.allclose(ret, adv + values)

def test_gae_done_masking():
    rewards = torch.tensor([[[1.0]], [[1.0]], [[1.0]]])
    values = torch.tensor([[[0.0]], [[0.0]], [[0.0]]])
    dones = torch.tensor([[[0.0]], [[1.0]], [[0.0]]])
    last_values = torch.tensor([[1.0]])

    adv, ret = utils.generalized_advantage_estimates(rewards, values, dones, last_values)

    assert adv[1].abs().sum() < adv[0].abs().sum(), "GAE should reset after done"

# Rewards to go tests
# ==================================================
def test_rtg_time_major():
    rewards = torch.tensor([[[1.0]], [[1.0]], [[1.0]]])
    dones = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])
    expected = torch.tensor([[[2.9701]], [[1.99]], [[1.0]]])
    rtg = utils.rewards_to_go(rewards, dones, gamma=0.99)
    assert torch.allclose(rtg, expected, atol=1e-4)

def test_rtg_batch_flat():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    expected = torch.tensor([[2.9701], [1.99], [1.0]])
    rtg = utils.rewards_to_go(rewards, dones, gamma=0.99)
    assert torch.allclose(rtg, expected, atol=1e-4)

def test_rtg_reset_on_done():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    dones = torch.tensor([[[0.0]], [[0.0]], [[1.0]], [[0.0]]])
    rtg = utils.rewards_to_go(rewards, dones, gamma=1.0)
    expected = torch.tensor([[[6.0]], [[5.0]], [[3.0]], [[4.0]]])
    assert torch.allclose(rtg, expected, atol=1e-4)

def test_rtg_multiple_envs():
    rewards = torch.tensor([[[1.0], [1.0]], [[1.0], [1.0]], [[1.0], [1.0]]])
    dones = torch.tensor([[[0.0], [0.0]], [[0.0], [1.0]], [[1.0], [0.0]]])
    rtg = utils.rewards_to_go(rewards, dones, gamma=1.0)
    expected = torch.tensor([[[3.0], [2.0]], [[2.0], [1.0]], [[1.0], [1.0]]])
    assert torch.allclose(rtg, expected, atol=1e-4)

def test_rtg_with_bootstrap_last_values():
    rewards = torch.tensor([[[1.0]], [[1.0]], [[1.0]]])
    dones = torch.tensor([[[0.0]], [[0.0]], [[0.0]]])
    last_values = torch.tensor([[10.0]])
    expected = torch.tensor([[[12.6731]], [[11.791]], [[10.9]]])  # gamma = 0.99
    rtg = utils.rewards_to_go(rewards, dones, gamma=0.99, last_values=last_values)
    assert torch.allclose(rtg, expected, atol=1e-4)

def test_rtg_with_last_values_and_dones():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    dones = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])  # done at t=2
    last_values = torch.tensor([[10.0]])
    expected = torch.tensor([[[5.9203]], [[4.97]], [[3.0]]])  # no bootstrap at done=1
    rtg = utils.rewards_to_go(rewards, dones, gamma=0.99, last_values=last_values)
    assert torch.allclose(rtg, expected, atol=1e-4)

def test_rtg_batch_flat_with_last_values():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.tensor([[0.0], [0.0], [0.0]])
    last_values = torch.tensor([[10.0]])
    expected = torch.tensor([[12.6731], [11.791], [10.9]])
    rtg = utils.rewards_to_go(rewards, dones, gamma=0.99, last_values=last_values)
    assert torch.allclose(rtg, expected, atol=1e-4)    

# Trajectory returns tests
# ==================================================
def test_trajectory_returns_terminal():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    dones = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])
    expected = torch.tensor([[[6.0]], [[6.0]], [[6.0]]])
    result = utils.trajectory_returns(rewards, dones, gamma=1.0)
    assert torch.allclose(result, expected, atol=1e-4)

def test_trajectory_returns_bootstrap():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    dones = torch.tensor([[[0.0]], [[0.0]], [[0.0]]])
    last_values = torch.tensor([[10.0]])
    expected = torch.tensor([[[16.0]], [[16.0]], [[16.0]]])
    result = utils.trajectory_returns(rewards, dones, last_values=last_values, gamma=1.0)
    assert torch.allclose(result, expected, atol=1e-4)

def test_trajectory_returns_gamma():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    dones = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])
    expected_scalar = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
    expected = torch.full((3, 1, 1), expected_scalar)
    result = utils.trajectory_returns(rewards, dones, gamma=0.99)
    assert torch.allclose(result, expected, atol=1e-3)

def test_trajectory_multiple_trajectories():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    dones = torch.tensor([[[0.0]], [[1.0]], [[0.0]], [[1.0]]])  # two trajectories
    expected = torch.tensor([[[3.0]], [[3.0]], [[7.0]], [[7.0]]])
    result = utils.trajectory_returns(rewards, dones, gamma=1.0)
    assert torch.allclose(result, expected, atol=1e-4)

def test_trajectory_returns_shape_error():
    rewards = torch.tensor([[[1.0]], [[2.0]]])
    dones = torch.tensor([[1.0], [0.0]])  # mismatched shape
    with pytest.raises(ValueError):
        utils.trajectory_returns(rewards, dones)        