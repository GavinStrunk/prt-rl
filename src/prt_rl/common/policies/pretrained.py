import torch
from torch import Tensor
from typing import Dict, Tuple

class SB3Policy:
    """
    Stable Baselines3 (SB3) agent for reinforcement learning. 
    
    This agent wraps a pre-trained model from Stable Baselines3 and uses it to predict actions based on the current state.

    Note:
        You must install prt-rl[sb3] to use this agent, which includes the necessary dependencies for Stable Baselines3.

    Args:
        model_dir (str): Path to the pre-trained model file.
        model_type (str): Type of the model (e.g., 'ppo', 'dqn', 'sac', etc.).
        device (str): Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
        **kwargs: Additional keyword arguments to pass to the model loading function.

    Reference:
        [1] [Model Library](https://huggingface.co/sb3)
    """
    def __init__(self, 
                 model_dir: str, 
                 model_type: str,
                 env_name: str,
                 device: str = "cpu",
                 **kwargs
                 ) -> None:
        self.device = torch.device(device)

        try:
            from sb3_contrib import ARS, QRDQN, TQC, TRPO, CrossQ, RecurrentPPO
            from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
            from stable_baselines3.common.base_class import BaseAlgorithm
            from rl_zoo3.load_from_hub import download_from_hub
            from rl_zoo3.utils import get_model_path

            ALGOS: dict[str, type[BaseAlgorithm]] = {
                "a2c": A2C,
                "ddpg": DDPG,
                "dqn": DQN,
                "ppo": PPO,
                "sac": SAC,
                "td3": TD3,
                # SB3 Contrib,
                "ars": ARS,
                "crossq": CrossQ,
                "qrdqn": QRDQN,
                "tqc": TQC,
                "trpo": TRPO,
                "ppo_lstm": RecurrentPPO,
            }
            if model_type.lower() not in ALGOS:
                raise ValueError(f"Model type '{model_type}' is not supported. Supported types are: {list(ALGOS.keys())}")
            
            self.model_type = model_type.lower()
            self.model_class = ALGOS[self.model_type]
        except ImportError:
            raise ImportError("Please install prt-rl with stable-baselines3 support with pip install prt-rl[sb3].")
        
        # Get the model path and if it does not exist, download it from the hub
        try:
            _, self.model_path, _ = get_model_path(exp_id=0, folder=model_dir, algo=self.model_type, env_name=env_name)
        except (AssertionError, ValueError) as e:
            download_from_hub(
                algo=self.model_type, 
                env_name=env_name, 
                exp_id=0,
                folder=model_dir,
                organization="sb3",
                repo_name=None,
                force=False 
                )
            _, self.model_path, _ = get_model_path(exp_id=0, folder=model_dir, algo=self.model_type, env_name=env_name)
        
        # Load the model
        self.model = self.model_class.load(self.model_path, device=self.device, **kwargs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Perform an action based on the current observation.

        Args:
            obs: The current observation of the environment.

        Returns:
            The action to be taken.
        """
        # Move state to model device
        obs = obs.to(self.device)

        # Wrap state in a batch if needed (for SB3, it expects NumPy)
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        action, _ = self.model.predict(obs, deterministic=deterministic)

        if len(action.shape) == 1:
            # Discrete actions are returned with shape (batch_size,)
            action = torch.tensor(action, device=self.device).unsqueeze(-1)
        else:
            action = torch.tensor(action, device=self.device)
        return action, {}
