from pathlib import Path
import torch
from torch import Tensor
from typing import Dict, List
from prt_rl.common.buffers import ReplayBuffer
from prt_rl.model_based.models.representation.keypoint import KeypointModel


class KeypointModelTrainer:
  def __init__(self, 
               keypoint_model: KeypointModel
               ) -> None:
    self.model = keypoint_model
    
  def train(self,
            replay_buffer: ReplayBuffer
            ):
    self._validate_replay_buffer(replay_buffer)

  def save(self, path: str | Path) -> None:
    pass

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
