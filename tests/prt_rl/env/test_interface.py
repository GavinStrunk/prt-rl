from tensordict.tensordict import TensorDict
import torch
from prt_rl.env.interface import EnvironmentInterface

def test_mdp_step():
    fake_mdp = TensorDict({
        'observation': torch.tensor([[8]]),
        'action': torch.tensor([[3]]),
        'next': {
            'observation': torch.tensor([[9]]),
            'done': torch.tensor([[False]]),
            'reward': torch.tensor([[-1.0]]),
        }
    },
    batch_size=torch.Size([1])
    )

    next_mdp = TensorDict({
        'observation': torch.tensor([[9]]),
    },
    batch_size=torch.Size([1])
    )

    # Step the MDP
    mdp = EnvironmentInterface.step_mdp(fake_mdp)

    # Check the Tensordicts are equal
    assert mdp.keys() == next_mdp.keys()
    for key in next_mdp.keys():
        assert torch.allclose(mdp[key], next_mdp[key])

def test_rgb_array_step():
    fake_mdp = TensorDict({
        'observation': torch.tensor([[8]]),
        'action': torch.tensor([[3]]),
        'next': {
            'observation': torch.tensor([[9]]),
            'done': torch.tensor([[False]]),
            'reward': torch.tensor([[-1.0]]),
            'rgb_array': torch.zeros((1, 100, 100, 3)),
        }
    },
    batch_size=torch.Size([1])
    )

    next_mdp = TensorDict({
        'observation': torch.tensor([[9]]),
        'rgb_array': torch.zeros((1, 100, 100, 3)),
    },
    batch_size=torch.Size([1])
    )

    # Step the MDP
    mdp = EnvironmentInterface.step_mdp(fake_mdp)

    # Check the Tensordicts are equal
    assert mdp.keys() == next_mdp.keys()
    for key in next_mdp.keys():
        assert torch.allclose(mdp[key], next_mdp[key])