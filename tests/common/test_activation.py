import pytest
import torch
from prt_rl.common.activation import TeLU

def test_telu_activation_at_zero():
    input = torch.zeros(1, 1)
    activation = TeLU()

    output = activation(input)
    assert output.shape == (1, 1)
    assert output[0, 0] == 0

def test_telu_positive_value():
    input = torch.ones(1,)
    activation = TeLU()
    output = activation(input)

    assert output.shape == (1,)
    assert output[0] == pytest.approx(0.9913, abs=1e-4)

def test_telu_negative_value():
    input = -torch.ones(1,)
    activation = TeLU()
    output = activation(input)

    assert output.shape == (1,)
    assert output[0] == pytest.approx(-0.3521, abs=1e-4)