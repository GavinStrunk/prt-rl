import torch
from typing import Optional, Tuple
from prt_rl.env.interface import EnvParams
from prt_rl.common.networks import MLP


class StateActionCritic(torch.nn.Module):
    """
    StateActionCritic is a critic network that takes both state and action as input and outputs the Q-value for the given state-action pair. It can handle multiple critics for ensemble methods.

    .. note::
        If multiple critics are used and an encoder is provided, the encoder will be shared across all critics. If no encoder is provided, the input state is used directly.

    The architecture of the critic is as follows:
        - Encoder Network (optional): Processes the input state.
        - Critic Head: Computes Q-values for the given state-action pair.

    .. image:: /_static/stateactioncritic.png
        :alt: StateActionCritic Architecture
        :width: 100%
        :align: center
    
    Args:
        env_params (EnvParams): Environment parameters.
        num_critics (int): Number of critics to use. Default is 1.
        encoder (torch.nn.Module | None): Encoder network to process the input state. If None, the input state is used directly.
        critic_head (torch.nn.Module): Critic head network to compute Q-values. Default is MLP.
        critic_head_kwargs (Optional[dict]): Keyword arguments for the critic head network.
    """
    def __init__(self, 
                 env_params: EnvParams, 
                 num_critics: int = 1,
                 encoder: torch.nn.Module | None = None,
                 critic_head: torch.nn.Module = MLP,
                 critic_head_kwargs: Optional[dict] = {},
                 ) -> None:
        super(StateActionCritic, self).__init__()
        self.env_params = env_params
        self.num_critics = num_critics
        self.encoder = encoder

        if self.encoder is not None:
            latent_dim = self.encoder.features_dim
        else:
            latent_dim = self.env_params.observation_shape[0]

        # Initialize critics here
        self.critics = []
        for _ in range(num_critics):
            critic = critic_head(
                input_dim=latent_dim + self.env_params.action_len,
                output_dim=1,
                **critic_head_kwargs
            )
            self.critics.append(critic)

        # Convert list to ModuleList for proper parameter management
        self.critics = torch.nn.ModuleList(self.critics)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor] | torch.Tensor:
        """
        Forward pass through the critic network.

        Args:
            state: The current state of the environment.
            action: The action taken in the current state.

        Returns:
            A tuple of Q-values for the given state-action pair for all critics.
        """
        if self.encoder is not None:
            state = self.encoder(state)

        # Stack the state and action tensors
        q_input = torch.cat([state, action], dim=1)

        # Return a tuple of Q-values from each critic or a single tensor if only one critic is used
        if self.num_critics == 1:
            return self.critics[0](q_input)
        else:
            return tuple(critic(q_input) for critic in self.critics)

    def forward_indexed(self, index: int, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network at the index provided.

        Args:
            index (int): The index of the critic to use.
            state (torch.Tensor): The current state of the environment.
            action (torch.Tensor): The action taken in the current state.

        Returns:
            The Q-value for the given state-action pair from the first critic.
        """
        if index > self.num_critics - 1:
            raise ValueError(f"Index {index} exceeds the number of critics {self.num_critics}.")
        
        if self.encoder is not None:
            state = self.encoder(state)

        # Stack the state and action tensors
        q_input = torch.cat([state, action], dim=1)
        return self.critics[index](q_input)