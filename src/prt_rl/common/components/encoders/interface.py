from torch import nn

class EncoderInterface(nn.Module):
    @property
    def latent_dim(self) -> int:
        raise NotImplementedError
    
    def forward(self, x: nn.Module) -> nn.Module:
        raise NotImplementedError