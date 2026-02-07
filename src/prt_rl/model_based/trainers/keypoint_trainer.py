from pathlib import Path
import torch
from torch import Tensor
from typing import Dict, List
from prt_rl.common.buffers import ReplayBuffer
from prt_rl.model_based.specs.representation import KeypointRepresentationSpec
from prt_rl.model_based.models.representation.factory import RepresentationModelFactory
from prt_rl.model_based.models.representation.keypoint import KeypointModel

class KeypointModelFactory(RepresentationModelFactory[KeypointRepresentationSpec, KeypointModel]):
    def make(self, env_params, spec):
        return KeypointModel(env_params=env_params, spec=spec)

    def save(self, env_params, spec, model, path):
        torch.save({
            'env_params': env_params,
            'spec': spec,
            'model_state_dict': model.state_dict(),
        }, path)

    def load(self, path, map_location="cpu"):
        data = torch.load(path, map_location=map_location)
        env_params = data['env_params']
        spec = data['spec']
        model = self.make(env_params, spec)
        model.load_state_dict(data['model_state_dict'])
        return model

class KeypointModelTrainer:
  def __init__(self, 
               keypoint_model_spec: KeypointRepresentationSpec
               ) -> None:
    self.model_spec = keypoint_model_spec
    
    # Make a KeypointModel based on the spec
    self.model = KeypointModelFactory().make(self.env_params, keypoint_model_spec)
    
  def train(self,
            replay_buffer: ReplayBuffer
            ):
    self._validate_replay_buffer(replay_buffer)

  def save(self, path: str | Path) -> None:
    KeypointModelFactory().save(self.env_params, self.model_spec, self.model, path)

  def _validate_replay_buffer(self, replay_buffer: ReplayBuffer) -> None:
    # Check the EnvParams are in the replay buffer metadata
    metadata = replay_buffer.get_metadata()
    if 'env_params' not in metadata:
      raise ValueError("Replay buffer metadata must contain 'env_params' key with environment parameters for training.")
    
    self.env_params = metadata['env_params']

    # Check if the replay buffer has the "episode_id" field and if not add it
    if "episode_id" not in replay_buffer.buffer:
      raise ValueError("Replay buffer must have 'episode_id' field for keypoint model training.")

  def compute_loss(self, x_t: Tensor, x_hat: Tensor) -> Dict[str, Tensor]:
    """
    Compute reconstruction loss between target and reconstructed images.

    Args:
        x_t: [B, C, H, W] target images
        x_hat: [B, C, H, W] reconstructed images
    Returns:
        dict of loss components
    """
    pass
