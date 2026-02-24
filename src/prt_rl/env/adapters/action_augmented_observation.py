import torch
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.env.adapters.interface import AdapterInterface

class ActionAugmentedObservationAdapter(AdapterInterface):
    """
    Adapter that augments the observation with the previous action taken. The observation is concatenated with the previous action along the last dimension. 

    Args:
        env (EnvironmentInterface): The environment to adapt
    """
    def __init__(self, 
                 env: EnvironmentInterface
                 ) -> None:
        params = env.get_parameters()
        if not params.action_continuous:
            raise ValueError("ActionAugmentedObservationAdapter only supports environments with continuous action spaces.")
        
        if len(params.observation_shape) != 1:
            raise ValueError("ActionAugmentedObservationAdapter only supports environments with 1D observation spaces.")
        
        self.action_dim = params.action_len
        self.previous_action = None
        super().__init__(env)

    def _adapt_params(self, params):
        # Update the observation shape to include action dimensions
        original_obs_dim = params.observation_shape[0]
        params.observation_shape = (original_obs_dim + self.action_dim,)

        if isinstance(params.observation_min, list):
            observation_min = params.observation_min
        else:
            observation_min = [params.observation_min] * original_obs_dim

        if isinstance(params.observation_max, list):
            observation_max = params.observation_max
        else:
            observation_max = [params.observation_max] * original_obs_dim

        if isinstance(params.action_min, list):
            action_min = params.action_min
        else:
            action_min = [params.action_min] * self.action_dim

        if isinstance(params.action_max, list):
            action_max = params.action_max
        else:
            action_max = [params.action_max] * self.action_dim

        params.observation_min = observation_min + action_min
        params.observation_max = observation_max + action_max
        return params
    
    def _adapt_action(self, action):
        """Store the previous action"""
        self.previous_action = action
        return super()._adapt_action(action)
    
    def _adapt_obs(self, obs, info):
        # Concatenate the previous action to the observation
        batch_size = obs.shape[0]
        if self.previous_action is None:
            # If no previous action, use zeros
            self.previous_action = torch.zeros((batch_size, self.action_dim), device=obs.device)
        
        augmented_obs = torch.cat([obs, self.previous_action], dim=-1)
        return augmented_obs
