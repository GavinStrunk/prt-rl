import torch
from typing import Optional
from prt_rl.env.interface import EnvParams
from prt_rl.common.networks import MLP

class ValueCritic(torch.nn.Module):
    """
    ValueCritic is a critic network that estimates the value of a given state.

    The architecture of the critic is as follows:
        - Encoder Network (optional): Processes the input state.
        - Critic Head: Computes the value for the given state.

    .. image:: /_static/valuecritic.png
        :alt: ValueCritic Architecture
        :width: 100%
        :align: center
    
    Args:
        env_params (EnvParams): Environment parameters.
        encoder (torch.nn.Module | None): Encoder network to process the input state. If None, the input state is used directly.
        critic_head (torch.nn.Module): Critic head network to compute values. Default is MLP.
        critic_head_kwargs (Optional[dict]): Keyword arguments for the critic head network.
    """
    def __init__(self,
                 env_params: EnvParams,
                 encoder: torch.nn.Module | None = None,
                 critic_head: torch.nn.Module = MLP,
                 critic_head_kwargs: Optional[dict] = {},                 
                 ) -> None:
        super().__init__()
        self.env_params = env_params
        self.encoder = encoder

        if self.encoder is not None:
            latent_dim = self.encoder.features_dim
        else:
            latent_dim = self.env_params.observation_shape[0]

        self.critic_head = critic_head(
            input_dim=latent_dim,
            output_dim=1,
            **critic_head_kwargs
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.

        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            torch.Tensor: The estimated value for the given state.
        """
        if self.encoder is not None:
            state = self.encoder(state)

        return self.critic_head(state)