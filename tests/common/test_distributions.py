from prt_rl.common.distributions import Categorical, Normal
import pytest
import torch
from torch.distributions import Categorical as TorchCategorical
from torch.distributions import Normal as TorchNormal

class DummyEnvParams:
    def __init__(self, action_len, action_min, action_max, action_continuous):
        self.action_len = action_len
        self.action_min = action_min
        self.action_max = action_max
        self.action_continuous = action_continuous

@pytest.fixture
def probs():
    # Shape: (batch, actions, 1)
    return torch.tensor([
        [[0.1], [0.2], [0.7]],
        [[0.3], [0.3], [0.4]]
    ])  # Shape (2, 3, 1)

def test_constructor_squeezes_last_dim(probs):
    dist = Categorical(probs)
    assert isinstance(dist, TorchCategorical)
    assert dist.probs.shape == (2, 3)

def test_deterministic_action(probs):
    dist = Categorical(probs)
    det_action = dist.deterministic_action()
    assert torch.equal(det_action, torch.tensor([2, 2]))  # argmax of each batch row

def test_sample_shape(probs):
    dist = Categorical(probs)
    samples = dist.sample()
    assert samples.shape == (2, 1)  # one sample per batch

def test_get_action_dim():
    env_params = DummyEnvParams(action_len=1, action_min=0, action_max=4, action_continuous=False)
    assert Categorical.get_action_dim(env_params) == 5

def test_get_action_dim_raises_for_continuous():
    env_params = DummyEnvParams(action_len=1, action_min=0, action_max=4, action_continuous=True)
    with pytest.raises(ValueError):
        Categorical.get_action_dim(env_params)

def test_last_network_layer_output_shape():
    feature_dim = 16
    action_dim = 5
    net = Categorical.last_network_layer(feature_dim, action_dim)

    x = torch.randn(4, feature_dim)
    out = net(x)
    assert out.shape == (4, action_dim)
    assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)  # softmax sum to 1

@pytest.fixture
def mean_tensor():
    # Shape: (batch, action_dim)
    return torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
    ])  # Shape (2, 3)

@pytest.fixture
def log_std_param():
    # Shape: (action_dim,)
    return torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))  # Shape (3,)

def test_constructor_shape_check():
    # Bad shape
    with pytest.raises(ValueError):
        _ = Normal(torch.randn(3, 3, 1), torch.nn.Parameter(torch.zeros(3)))

def test_constructor_creates_distribution(mean_tensor, log_std_param):
    dist = Normal(mean_tensor, log_std_param)
    assert isinstance(dist, TorchNormal)
    assert dist.mean.shape == (2, 3)
    assert dist.stddev.shape == (2, 3)  

def test_deterministic_action_returns_mean(mean_tensor, log_std_param):
    dist = Normal(mean_tensor, log_std_param)
    action = dist.deterministic_action()
    assert torch.equal(action, mean_tensor)

def test_sample_shape(mean_tensor, log_std_param):
    dist = Normal(mean_tensor, log_std_param)
    samples = dist.sample()
    assert samples.shape == mean_tensor.shape

def test_get_action_dim():
    env_params = DummyEnvParams(action_len=6, action_min=0, action_max=1, action_continuous=True)
    assert Normal.get_action_dim(env_params) == 6

def test_get_action_dim_raises_for_discrete():
    env_params = DummyEnvParams(action_len=1, action_min=0, action_max=5, action_continuous=False)
    with pytest.raises(ValueError):
        Normal.get_action_dim(env_params)

def test_last_network_layer_output_and_log_std():
    feature_dim = 8
    action_dim = 4
    log_std_init = -0.5

    layer, log_std = Normal.last_network_layer(feature_dim, action_dim, log_std_init)

    assert isinstance(layer, torch.nn.Linear)
    assert layer.in_features == feature_dim
    assert layer.out_features == action_dim

    assert isinstance(log_std, torch.nn.Parameter)
    assert torch.allclose(log_std.data, torch.full((action_dim,), log_std_init))
    assert log_std.requires_grad