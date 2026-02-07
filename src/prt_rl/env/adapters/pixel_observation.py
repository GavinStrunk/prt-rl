import torch
from typing import Literal
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.env.adapters.interface import AdapterInterface

class PixelObservationAdapter(AdapterInterface):
    """
    Adapater takes the 'rgb_array' pixels from the info dictionary and makes that the observation. The original observation is added back into the info dictionary under the 'state' key.

    Assumes pixel observation are [H, W, C] numpy arrays of type uint8 and converts to [C, H, W] torch tensors normalized to [0, 1].
    Note: This adapter assumes the base environment has render_mode set to "rgb_array".

    Args:
        env (EnvironmentInterface): The environment to adapt
    """
    def __init__(self, 
                 env: EnvironmentInterface,
                 pixel_type: Literal["uint8", "float32"] = "uint8"
                 ):
        self.pixel_type = pixel_type
        _, info = env.reset()
        self.image_shape = info['rgb_array'][0].shape
        super().__init__(env)

    def _adapt_params(self, params):
        # Update the observation shape with the image shape
        params.observation_shape = self.image_shape
        params.observation_continuous = True
        params.observation_min = 0.0
        params.observation_max = 1.0
        return params
    
    def _adapt_info(self, action, obs, reward, done, info):
        # Remove 'rgb_array' and add 'state' from the observation
        new_info = {k: v for k, v in info.items() if k != 'rgb_array'}
        new_info['state'] = obs
        return new_info
    
    def _adapt_obs(self, obs, info):
        pixels = torch.from_numpy(info['rgb_array'][0]).permute(2, 0, 1)  # [C, H, W]
        pixels = pixels.unsqueeze(0).to(obs.device)  # add batch dimension back in

        if self.pixel_type == "float32":
            pixels = pixels.to(torch.float32) / 255.0  # normalize to [0, 1]
        return pixels