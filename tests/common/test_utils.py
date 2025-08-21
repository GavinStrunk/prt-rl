import torch
import torch.nn as nn
import pytest
from copy import deepcopy
import prt_rl.common.utils as utils

# Clamp Actions tests
# ==================================================
def test_clamp_scalar_bounds():
    actions = torch.tensor([[2.0, -3.0], [0.5, 1.5]])
    clamped = utils.clamp_actions(actions, -1.0, 1.0)
    expected = torch.tensor([[1.0, -1.0], [0.5, 1.0]])
    assert torch.allclose(clamped, expected)


def test_clamp_list_bounds():
    actions = torch.tensor([[2.0, -3.0], [0.5, 1.5]])
    clamped = utils.clamp_actions(actions, [-1.0, -2.0], [1.0, 2.0])
    expected = torch.tensor([[1.0, -2.0], [0.5, 1.5]])
    assert torch.allclose(clamped, expected)


def test_clamp_tensor_bounds():
    actions = torch.tensor([[2.0, -3.0], [0.5, 1.5]])
    action_min = torch.tensor([-1.0, -2.0])
    action_max = torch.tensor([1.0, 2.0])
    clamped = utils.clamp_actions(actions, action_min, action_max)
    expected = torch.tensor([[1.0, -2.0], [0.5, 1.5]])
    assert torch.allclose(clamped, expected)


def test_clamp_no_clipping():
    actions = torch.tensor([[0.0, 0.5], [0.2, -0.5]])
    clamped = utils.clamp_actions(actions, -1.0, 1.0)
    assert torch.allclose(clamped, actions)


def test_clamp_on_cuda_if_available():
    if torch.cuda.is_available():
        actions = torch.tensor([[2.0, -3.0]], device='cuda')
        clamped = utils.clamp_actions(actions, -1.0, 1.0)
        expected = torch.tensor([[1.0, -1.0]], device='cuda')
        assert torch.allclose(clamped, expected)


def test_clamp_invalid_length_list():
    actions = torch.randn(2, 3)
    with pytest.raises(RuntimeError):
        utils.clamp_actions(actions, [-1.0, 0.0], [1.0, 1.0])  # Wrong length


def test_clamp_invalid_length_tensor():
    actions = torch.randn(2, 3)
    min_tensor = torch.tensor([-1.0, 0.0])  # Invalid
    max_tensor = torch.tensor([1.0, 1.0])   # Invalid
    with pytest.raises(RuntimeError):
        utils.clamp_actions(actions, min_tensor, max_tensor)

def test_clamp_discrete_actions_fails():
    actions = torch.tensor([[2], [1]], dtype=torch.int32)
    with pytest.raises(ValueError):
        utils.clamp_actions(actions, -1, 1)

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

# Normalized advantages tests
# ==================================================
def test_normalize_advantages_zero_mean_unit_std():
    advantages = torch.tensor([[1.0], [2.0], [3.0]])
    normalized = utils.normalize_advantages(advantages)
    
    # Check shape
    assert advantages.shape == (3, 1)
    assert normalized.shape == advantages.shape

    # Check mean ~ 0 and std ~ 1
    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized.std(), torch.tensor(1.0), atol=1e-6)

def test_normalize_advantages_identical_values():
    advantages = torch.tensor([[5.0], [5.0], [5.0]])
    normalized = utils.normalize_advantages(advantages)

    # All outputs should be 0 when all inputs are the same (std is zero)
    expected = torch.zeros_like(advantages)
    torch.testing.assert_close(normalized, expected)

def test_normalize_advantages_single_element():
    advantages = torch.tensor([[10.0]])
    normalized = utils.normalize_advantages(advantages)

    # Single value minus mean (same value) divided by std (zero) + epsilon => 0
    print(normalized)
    expected = torch.tensor([[0.0]])
    torch.testing.assert_close(normalized, expected)

def test_normalize_advantages_batch_shape():
    # Shape (B, 1) with B > 3
    advantages = torch.randn(100, 1)
    normalized = utils.normalize_advantages(advantages)

    assert normalized.shape == (100, 1)
    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized.std(), torch.tensor(1.0), atol=1e-6)

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

# Gaussian noise tests
# ==================================================

def test_gaussian_noise_shape():
    shape = (3, 4)
    noise = utils.gaussian_noise(mean=0.0, std=1.0, shape=shape)
    assert noise.shape == shape

def test_gaussian_noise_mean_std():
    shape = (10000,)
    mean = 2.0
    std = 3.0
    noise = utils.gaussian_noise(mean=mean, std=std, shape=shape)
    assert abs(noise.mean().item() - mean) < 0.1
    assert abs(noise.std(unbiased=True).item() - std) < 0.1

def test_gaussian_noise_zero_std():
    shape = (5, 5)
    mean = 1.23
    std = 0.0
    noise = utils.gaussian_noise(mean=mean, std=std, shape=shape)
    expected = torch.full(shape, mean)
    assert torch.allclose(noise, expected, atol=1e-6)

def test_gaussian_noise_default():
    noise = utils.gaussian_noise()
    assert isinstance(noise, torch.Tensor)
    assert noise.shape == (1,)

# Ornstein-Uhlenbeck process tests
# ==================================================
def test_ou_noise_shape():
    shape = (3, 4)
    noise = utils.ornstein_uhlenbeck_noise(mean=0.0, std=1.0, shape=shape)
    assert noise.shape == shape

def test_ou_noise_zero_std():
    shape = (2, 2)
    x0 = torch.ones(shape)
    mean = 0.5
    std = 0.0
    theta = 0.2
    dt = 0.1
    expected = x0 + theta * (mean - x0) * dt
    noise = utils.ornstein_uhlenbeck_noise(mean=mean, std=std, shape=shape, theta=theta, dt=dt, x0=x0)
    assert torch.allclose(noise, expected, atol=1e-6)

def test_ou_noise_deterministic_seed():
    torch.manual_seed(42)
    n1 = utils.ornstein_uhlenbeck_noise(shape=(5,))
    torch.manual_seed(42)
    n2 = utils.ornstein_uhlenbeck_noise(shape=(5,))
    assert torch.allclose(n1, n2)

def test_ou_noise_strong_theta_pull():
    shape = (3,)
    x0 = torch.tensor([2.0, -2.0, 0.5])
    mean = 0.0
    theta_small = 0.01
    theta_large = 0.5
    dt = 0.1
    std = 0.0  # no stochasticity, test only drift
    noise_small = utils.ornstein_uhlenbeck_noise(mean=mean, std=std, shape=shape, theta=theta_small, dt=dt, x0=x0)
    noise_large = utils.ornstein_uhlenbeck_noise(mean=mean, std=std, shape=shape, theta=theta_large, dt=dt, x0=x0)
    assert torch.norm(noise_large) < torch.norm(noise_small)

def test_ou_sequential_calls():
    shape = (2, 2)
    x0 = torch.zeros(shape)
    mean = 0.0
    std = 1.0
    theta = 0.15
    dt = 0.1

    torch.manual_seed(0)
    noise1 = utils.ornstein_uhlenbeck_noise(mean=mean, std=std, shape=shape, theta=theta, dt=dt, x0=x0)
    torch.manual_seed(0)
    noise2 = utils.ornstein_uhlenbeck_noise(mean=mean, std=std, shape=shape, theta=theta, dt=dt, x0=noise1)

    assert torch.allclose(noise1, torch.tensor([[0.4873, -0.0928], [-0.6890,  0.1798]]), atol=1e-4)
    assert torch.allclose(noise2, torch.tensor([[0.9673, -0.1842], [-1.3677,  0.3568]]), atol=1e-4)