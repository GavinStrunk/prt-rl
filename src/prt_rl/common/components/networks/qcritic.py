from torch import nn

class QCritic(nn.Module):
    """
    QCritic is a neural network module for estimating Q-values in reinforcement learning.

    This class composes a feature extraction network and a critic head to compute Q-values
    given observations and actions. It is typically used in actor-critic or value-based RL algorithms.

    Args:
        network (nn.Module): Feature extractor network that processes observations.
        critic_head (nn.Module): Head network that takes features and actions to output Q-values.
    """
    def __init__(self, network: nn.Module, critic_head: nn.Module):
        """
        Initialize the QCritic module.

        Args:
            network (nn.Module): Feature extractor for observations.
            critic_head (nn.Module): Module that computes Q-values from features and actions.
        """
        super().__init__()
        self.network = network
        self.critic_head = critic_head

    def forward(self, obs, action):
        """
        Forward pass to compute Q-values from observations and actions.

        Args:
            obs: Input observations (tensor or compatible type for network).
            action: Actions to evaluate (tensor or compatible type for critic_head).

        Returns:
            Q-values estimated by the critic (tensor).
        """
        features = self.network(obs)
        q = self.critic_head(features, action)   # adjust signature to your head
        return q