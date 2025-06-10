import torch
from prt_rl.dqn import DQN

class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        # Set known weights for reproducibility
        torch.nn.init.constant_(self.linear.weight, 1.0)
        torch.nn.init.constant_(self.linear.bias, 2.0)

def test_polyak_update():
    policy = DummyNet()
    target = DummyNet()
    tau = 0.5
    
    # Set target weights to a different known value
    torch.nn.init.constant_(target.linear.weight, 0.0)
    torch.nn.init.constant_(target.linear.bias, 0.0)

    DQN._polyak_update(policy, target, tau)
    assert torch.allclose(target.linear.weight, torch.full_like(target.linear.weight, 0.5))
    assert torch.allclose(target.linear.bias, torch.full_like(target.linear.bias, 1.0))

