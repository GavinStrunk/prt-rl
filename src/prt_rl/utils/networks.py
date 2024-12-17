import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, state_size, action_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class MLPSoftmax(nn.Module):
    def __init__(self, state_size, action_size):
        super(MLPSoftmax, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x