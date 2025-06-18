import pytest
from prt_rl.common.networks import *

def test_mlp_construction():
    # Default architecture is:
    # MLP(
    #   (0): Linear(in_features=4, out_features=64, bias=True)
    #   (1): ReLU()
    #   (2): Linear(in_features=64, out_features=64, bias=True)
    #   (3): ReLU()
    #   (4): Linear(in_features=64, out_features=3, bias=True)
    # )
    mlp = MLP(input_dim=4, output_dim=3)
    assert len(mlp.layers) == 5

    # Create network
    # MLP(
    #   (0): Linear(in_features=4, out_features=128, bias=True)
    #   (1): ReLU()
    #   (2): Linear(in_features=128, out_features=3, bias=True)
    # )
    mlp = MLP(input_dim=4, output_dim=3, network_arch=[128])
    assert len(mlp.layers) == 3

    # Set the hidden layer activation
    mlp = MLP(input_dim=4, output_dim=3, hidden_activation=nn.Tanh())
    assert isinstance(mlp.layers[1], nn.Tanh)

    # Set final layer activation
    mlp = MLP(input_dim=4, output_dim=3, final_activation=nn.Softmax(dim=-1))
    assert isinstance(mlp.final_activation, nn.Softmax)

def test_mlp_forward():
    mlp = MLP(input_dim=1, output_dim=2)
    state = torch.tensor([[1]], dtype=torch.float32)
    assert state.shape == (1, 1)
    qval = mlp(state)
    assert qval.shape == (1, 2)

def test_mlp_output_and_arch_cannot_be_none():
    with pytest.raises(ValueError):
        MLP(input_dim=4, output_dim=None, network_arch=None)

def test_mlp_no_output_dim():
    mlp = MLP(input_dim=4)
    assert len(mlp.layers) == 4
    assert mlp.layers[0].in_features == 4
    assert mlp.layers[0].out_features == 64
    assert isinstance(mlp.layers[1], nn.ReLU)
    assert mlp.layers[2].in_features == 64
    assert mlp.layers[2].out_features == 64
    assert isinstance(mlp.layers[3], nn.ReLU)

def test_dueling_mlp_construction():
    mlp = DuelingMLP(input_dim=512, output_dim=4)

    assert len(mlp.advantage_layers) == 1
    assert mlp.advantage_layers[0].in_features == 512
    assert mlp.advantage_layers[0].out_features == 4
    assert len(mlp.value_layers) == 1
    assert mlp.value_layers[0].in_features == 512
    assert mlp.value_layers[0].out_features == 1

def test_nature_cnn_encoder_construction():
    encoder = NatureCNNEncoder(input_shape=(4, 84, 84), features_dim=512)
    assert len(encoder.layers) == 9
    assert encoder.layers[0].in_channels == 4
    assert encoder.layers[0].out_channels == 32
    assert isinstance(encoder.layers[1], nn.ReLU)
    assert encoder.layers[2].in_channels == 32
    assert encoder.layers[2].out_channels == 64
    assert isinstance(encoder.layers[3], nn.ReLU)
    assert encoder.layers[4].in_channels == 64
    assert encoder.layers[4].out_channels == 64
    assert isinstance(encoder.layers[5], nn.ReLU)
    assert isinstance(encoder.layers[6], nn.Flatten)
    assert encoder.layers[7].in_features == 64 * 7 * 7
    assert encoder.layers[7].out_features == 512
    assert isinstance(encoder.layers[8], nn.ReLU)

def test_dueling_mlp_arch_construction():
    mlp = DuelingMLP(input_dim=512, output_dim=4, network_arch=[128, 64], hidden_activation=nn.Tanh(), final_activation=nn.Softmax(dim=-1))
    assert len(mlp.advantage_layers) == 6
    assert mlp.advantage_layers[0].in_features == 512
    assert mlp.advantage_layers[0].out_features == 128
    assert isinstance(mlp.advantage_layers[1], nn.Tanh)
    assert mlp.advantage_layers[2].in_features == 128
    assert mlp.advantage_layers[2].out_features == 64
    assert isinstance(mlp.advantage_layers[3], nn.Tanh)
    assert mlp.advantage_layers[4].in_features == 64
    assert mlp.advantage_layers[4].out_features == 4
    assert isinstance(mlp.advantage_layers[5], nn.Softmax)

    assert len(mlp.value_layers) == 6
    assert mlp.value_layers[0].in_features == 512
    assert mlp.value_layers[0].out_features == 128
    assert isinstance(mlp.value_layers[1], nn.Tanh)
    assert mlp.value_layers[2].in_features == 128
    assert mlp.value_layers[2].out_features == 64
    assert isinstance(mlp.value_layers[3], nn.Tanh)
    assert mlp.value_layers[4].in_features == 64
    assert mlp.value_layers[4].out_features == 1
    assert isinstance(mlp.value_layers[5], nn.Softmax)

def test_naturecnn_construction():
    cnn = NatureCNN(state_shape=(4, 84, 84), action_len=4)
    print(cnn)

    # Dummy input
    state = torch.rand(size=(1, 4, 84, 84), dtype=torch.float32)
    action = cnn(state)
    assert action.shape == (1, 4)

    # Batch dummy input
    state = torch.rand(size=(10, 4, 84, 84), dtype=torch.float32)
    action = cnn(state)
    assert action.shape == (10, 4)

def test_naturecnn_with_uint8():
    # Test with uint8 input
    cnn = NatureCNN(state_shape=(4, 84, 84), action_len=4)
    state = torch.randint(low=0, high=255, size=(1, 4, 84, 84), dtype=torch.uint8)
    action = cnn(state)
    assert action.shape == (1, 4)

def test_dueling_naturecnn():
    cnn = NatureCNN(state_shape=(4, 84, 84), action_len=4, dueling=True)
    state = torch.randint(low=0, high=255, size=(1, 4, 84, 84), dtype=torch.uint8)
    action = cnn(state)
    assert action.shape == (1, 4)