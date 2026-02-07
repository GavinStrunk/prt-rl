from torch import nn, Tensor

class RepresentationInterface(nn.Module):
    """
    Interface for representation models that encode observations into latent representations.
    """
    def encode(self, obs: Tensor) -> Tensor:
        """
        Encode the observation into a latent representation.

        Args:
            obs: input observation tensor
        Returns:
            latent representation tensor
        """
        raise NotImplementedError
    