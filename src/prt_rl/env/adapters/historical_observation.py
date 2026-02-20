import torch
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.env.adapters.interface import AdapterInterface


class HistoricalObservationAdapter(AdapterInterface):
    """
    Adapter that augments observations with a fixed-length observation history and optional action history.

    Args:
        env (EnvironmentInterface): The environment to adapt
        num_steps (int): Number of observations to stack in the augmented observation.
        include_actions (bool): If True, include previous actions between stacked observations.
        append_last_action (bool): If True and include_actions is True, append the most recent
            action to the end of the observation stack.
            Example (num_steps=3):
            - False: [o_{t-2}, o_{t-1}, o_t]
            - True:  [o_{t-2}, a_{t-2}, o_{t-1}, a_{t-1}, o_t]
            - True + append_last_action=True:
              [o_{t-2}, a_{t-2}, o_{t-1}, a_{t-1}, o_t, a_{t-1}]
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 num_steps: int = 4,
                 include_actions: bool = True,
                 append_last_action: bool = False,
                 ) -> None:
        params = env.get_parameters()

        if len(params.observation_shape) != 1:
            raise ValueError("HistoricalObservationAdapter only supports environments with 1D observation spaces.")

        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        if include_actions and not params.action_continuous:
            raise ValueError("HistoricalObservationAdapter with include_actions=True only supports continuous action spaces.")

        self.num_steps = num_steps
        self.include_actions = include_actions
        self.append_last_action = append_last_action
        self.action_dim = params.action_len
        self.observation_history = []
        self.action_history = []
        self.last_action = None
        super().__init__(env)

    def _adapt_params(self, params):
        original_obs_dim = params.observation_shape[0]
        num_action_slots = max(self.num_steps - 1, 0) if self.include_actions else 0
        if self.include_actions and self.append_last_action:
            num_action_slots += 1
        adapted_obs_dim = original_obs_dim * self.num_steps + self.action_dim * num_action_slots
        params.observation_shape = (adapted_obs_dim,)

        if isinstance(params.observation_min, list):
            observation_min = params.observation_min
        else:
            observation_min = [params.observation_min] * original_obs_dim

        if isinstance(params.observation_max, list):
            observation_max = params.observation_max
        else:
            observation_max = [params.observation_max] * original_obs_dim

        if len(observation_min) != original_obs_dim:
            raise ValueError(f"Expected observation_min length {original_obs_dim}, got {len(observation_min)}.")
        if len(observation_max) != original_obs_dim:
            raise ValueError(f"Expected observation_max length {original_obs_dim}, got {len(observation_max)}.")

        if self.include_actions:
            if isinstance(params.action_min, list):
                action_min = params.action_min
            else:
                action_min = [params.action_min] * self.action_dim

            if isinstance(params.action_max, list):
                action_max = params.action_max
            else:
                action_max = [params.action_max] * self.action_dim

            if len(action_min) != self.action_dim:
                raise ValueError(f"Expected action_min length {self.action_dim}, got {len(action_min)}.")
            if len(action_max) != self.action_dim:
                raise ValueError(f"Expected action_max length {self.action_dim}, got {len(action_max)}.")

            adapted_min = []
            adapted_max = []
            for idx in range(self.num_steps):
                adapted_min.extend(observation_min)
                adapted_max.extend(observation_max)
                if idx < self.num_steps - 1:
                    adapted_min.extend(action_min)
                    adapted_max.extend(action_max)
            if self.append_last_action:
                adapted_min.extend(action_min)
                adapted_max.extend(action_max)
        else:
            adapted_min = observation_min * self.num_steps
            adapted_max = observation_max * self.num_steps

        params.observation_min = adapted_min
        params.observation_max = adapted_max

        return params

    def reset(self, *args, **kwargs):
        # Clear temporal buffers at episode reset.
        self.observation_history = []
        self.action_history = []
        self.last_action = None
        return super().reset(*args, **kwargs)

    def _adapt_action(self, action):
        """Store the previous action."""
        if self.include_actions:
            self.last_action = action
            self.action_history.append(action)
            if len(self.action_history) > self.num_steps - 1:
                self.action_history.pop(0)
        return super()._adapt_action(action)

    def _adapt_obs(self, obs, info):
        """Store the current observation and return the stacked history."""
        self.observation_history.append(obs)
        if len(self.observation_history) > self.num_steps:
            self.observation_history.pop(0)

        padded_obs_history = [torch.zeros_like(obs)] * (self.num_steps - len(self.observation_history)) + self.observation_history
        if not self.include_actions:
            return torch.cat(padded_obs_history, dim=-1)

        batch_size = obs.shape[0]
        padded_action_history = [torch.zeros((batch_size, self.action_dim), dtype=obs.dtype, device=obs.device)] * (self.num_steps - 1 - len(self.action_history)) + self.action_history

        parts = []
        for idx, hist_obs in enumerate(padded_obs_history):
            parts.append(hist_obs)
            if idx < self.num_steps - 1:
                parts.append(padded_action_history[idx])

        if self.append_last_action:
            if self.last_action is None:
                last_action = torch.zeros((batch_size, self.action_dim), dtype=obs.dtype, device=obs.device)
            else:
                last_action = self.last_action
            parts.append(last_action)

        augmented_obs = torch.cat(parts, dim=-1)
        return augmented_obs
