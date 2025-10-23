import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any
from prt_rl.env.interface import EnvParams, EnvironmentInterface


class JhuWrapper(EnvironmentInterface):
    """
    Wraps the JHU environments in the Environment interface.

    The JHU environments are games and puzzles that were used in the JHU 705.741 RL course.

    Args:
        jhu_name (str): JHU Environment name
        env_args (dict): Arguments to pass to the JHU environment constructor
        render_mode (str, optional): Sets the render mode ['human', 'rgb_array']. Default: None.

    Examples:
        ```python
        from prt_sim.jhu.bandits import KArmBandits
        from prt_rl.env.wrappers import JhuWrapper
        from prt_rl.common.policy import RandomPolicy

        env = JhuWrapper(environment=KArmBandits())
        policy = RandomPolicy(env_params=env.get_parameters())

        state = env.reset(seed=0)
        done = False

        while not done:
            action = policy.get_action(state)
            next_state, reward, done, info = env.step(action)

        ```
    """

    def __init__(self,
                 jhu_name: str,
                 *,
                 render_mode: Optional[str] = None,
                 device: str = 'cpu',
                 **kwargs
                 ) -> None:
        super().__init__(render_mode)
        try:
            import prt_sim.jhu as jhu
        except ImportError as e:
            raise ImportError("prt-sim module is required for JhuWrapper. Please install prt-sim package.")
        
        self.jhu_name = jhu_name
        self.env = jhu.make(jhu_name, **kwargs)
        self.device = device

    def get_parameters(self) -> EnvParams:
        """
        Returns the EnvParams object which contains information about the sizes of observations and actions needed for setting up RL agents.
        Returns:
            EnvParams: environment parameters object
        """
        params = EnvParams(
            action_len=1,
            action_continuous=False,
            action_min=0,
            action_max=self.env.get_number_of_actions() - 1,
            observation_shape=(1,),
            observation_continuous=False,
            observation_min=0,
            observation_max=max(self.env.get_number_of_states() - 1, 0),
        )
        return params

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Resets the environment to the initial state and returns the initial observation.

        Args:
            seed (int | None): Sets the random seed.

        Returns:
            Tuple: Tuple of tensors containing the initial observation and info dictionary
        """
        info = {}
        state = self.env.reset(seed=seed)
        state = torch.tensor([[state]], dtype=torch.int64, device=self.device)

        # Add info for Bandit environment
        if 'KArmBandits-v0' in self.jhu_name:
            info = {
                'optimal_bandit': torch.tensor([[self.env.get_optimal_bandit()]], dtype=torch.int64, device=self.device),
                'bandits': torch.tensor([self.env.bandit_probs], dtype=torch.float32, device=self.device)
            }

        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            rgb = self.env.render()
            info['rgb_array'] = rgb[np.newaxis, ...]

        return state, info

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Steps the simulation using the action tensor and returns the new trajectory.

        Args:
            action (torch.Tensor): Tensor with "action" key that is a tensor with shape (# env, # actions)

        Returns:
            Tuple: Tuple of tensors containing the next state, reward, done, and info dictionary
        """
        info = {}

        # Convert tensor to numpy array
        action = action.cpu().numpy()
        state, reward, done = self.env.execute_action(action[0][0])

        # Convert integers to numpy arrays
        state = torch.tensor([[state]], dtype=torch.int64, device=self.device)
        reward = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        done = torch.tensor([[done]], dtype=torch.bool, device=self.device)

        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            rgb = self.env.render()
            info['rgb_array'] = rgb[np.newaxis, ...]

        return state, reward, done, info