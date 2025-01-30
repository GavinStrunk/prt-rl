"""
Custom pytorch activation functions for neural networks.

"""
import torch

class TeLU(torch.nn.Module):
    """
    Hyperbolic Tangent Exponential Linear Unit (TeLU)

    .. math::
        TeLU(x) = x tanh(e_x)

    .. image:: /_static/telu.png
        :alt: TeLU Activation Function
        :width: 100%
        :align: center

    References:
        [1] https://arxiv.org/pdf/2412.20269
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(TeLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.tanh(torch.exp(input))

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str