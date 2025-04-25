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
    mlp = MLP(state_dim=4, action_dim=3)
    assert len(mlp) == 5

    # Create network
    # MLP(
    #   (0): Linear(in_features=4, out_features=128, bias=True)
    #   (1): ReLU()
    #   (2): Linear(in_features=128, out_features=3, bias=True)
    # )
    mlp = MLP(state_dim=4, action_dim=3, network_arch=[128])
    assert len(mlp) == 3

    # Set the hidden layer activation
    mlp = MLP(state_dim=4, action_dim=3, hidden_activation=nn.Tanh())
    assert isinstance(mlp[1], nn.Tanh)

    # Set final layer activation
    mlp = MLP(state_dim=4, action_dim=3, final_activation=nn.Softmax(dim=-1))
    assert isinstance(mlp[-1], nn.Softmax)

def test_mlp_forward():
    mlp = MLP(state_dim=1, action_dim=2)
    state = torch.tensor([[1]], dtype=torch.float32)
    assert state.shape == (1, 1)
    qval = mlp(state)
    assert qval.shape == (1, 2)

def test_naturecnn_construction():
    cnn = NatureCNN(state_shape=(4, 84, 84), action_len=4)
    print(cnn)

    # Default architecture has 9 layers
    assert len(cnn) == 10

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