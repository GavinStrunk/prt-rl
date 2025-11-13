"""
Wrapper for Isaac Lab environments
"""
import torch
from typing import Optional, Tuple, List, Union, Dict, Any
from prt_rl.env.interface import EnvironmentInterface, EnvParams

import argparse
import sys
import atexit
import gymnasium
from prt_rl.env.wrappers.gymnasium_envs import GymnasiumWrapper

class IsaaclabWrapper(EnvironmentInterface):
    """

    """
    def __init__(self,
                 env_name: str,
                 render_mode: Optional[str] = None,
                 num_envs: int = 1,
                 headless: bool = True
                ) -> None:
        super().__init__(render_mode, num_envs=num_envs)

        # Add arguments to system arguments to create a parser
        sys.argv.append("--task")
        sys.argv.append(env_name)
        sys.argv.append("--num_envs")
        sys.argv.append(str(num_envs))

        if headless:
            sys.argv.append("--headless")

        # Create argument parsing object
        parser = argparse.ArgumentParser("Isaac Lab")
        parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
        parser.add_argument("--task", type=str, default=None, help="Name of the task")
        parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
        parser.add_argument(
            "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations"
        )
        parser.add_argument(
            "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes"
        )

        # launch the simulation app
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        app_launcher = AppLauncher(args)

        @atexit.register
        def close_the_simulator():
            app_launcher.app.close()

        import isaaclab_tasks  # type: ignore
        from isaaclab_tasks.utils import parse_env_cfg  # type: ignore

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
        if args.distributed:
            cfg.sim.device = f"cuda:{app_launcher.local_rank}"

        # load environment
        self.env = gymnasium.make(env_name, cfg=cfg)  

        # Create environment parameter object  
        self.env_params = self._make_env_params()

        self.first_reset = True
        self.state = None
        self.info = {}

    def _make_env_params(self) -> EnvParams:
        """
        Creates the environment parameters based on the action and observation space of the environment.
        Args:
            vectorized (bool): If True, the environment is vectorized.
        Returns:
            EnvParams: The environment parameters object.
        """
        action_space = self.env.unwrapped.single_action_space
        observation_space = self.env.unwrapped.single_observation_space["policy"]

        if isinstance(action_space, gymnasium.spaces.Discrete):
            action_len, act_cont, act_min, act_max = GymnasiumWrapper._get_params_from_discrete(action_space, is_action=True)
        elif isinstance(action_space, gymnasium.spaces.Box):
            action_len, act_cont, act_min, act_max = GymnasiumWrapper._get_params_from_box(action_space, is_action=True)
        else:
            raise NotImplementedError(f"{action_space} action space is not supported")

        if isinstance(observation_space, gymnasium.spaces.Discrete):
            obs_shape, obs_cont, obs_min, obs_max = GymnasiumWrapper._get_params_from_discrete(observation_space)
        elif isinstance(observation_space, gymnasium.spaces.Box):
            obs_shape, obs_cont, obs_min, obs_max = GymnasiumWrapper._get_params_from_box(observation_space)
        else:
            raise NotImplementedError(f"{observation_space} observation space is not supported")

        return EnvParams(
            action_len=action_len,
            action_continuous=act_cont,
            action_min=act_min,
            action_max=act_max,
            observation_shape=obs_shape,
            observation_continuous=obs_cont,
            observation_min=obs_min,
            observation_max=obs_max,
        )        

    def get_parameters(self) -> EnvParams:
        """
        Returns the EnvParams object which contains information about the sizes of observations and actions needed for setting up RL agents.

        Returns:
            EnvParams: environment parameters object
        """
        return self.env_params

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Resets the environment to the initial state and returns the initial observation.

        Args:
            seed (int | None): Sets the random seed.

        Returns:
            Tuple: Tuple of tensors containing the initial observation and info dictionary
        """
        if self.first_reset:
            state, self.info = self.env.reset(seed=seed)
            self.state = state['policy']
            self.first_reset = False

        # The state is a dictionary and the observation is in the key 'policy'. Sometimes there is also a 'critic' key for separate actor/critic observations.
        return self.state, self.info
    
    def reset_index(self, index: int, seed: int | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Resets only the environments that are done.

        Args:
            done (torch.Tensor): Boolean tensor of shape (num_envs, 1) or (num_envs,)

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The new observations and info dict
        """
        if self.first_reset:
            state, self.info = self.env.reset(seed=seed)
            self.state = state['policy']
            self.first_reset = False

        return self.state[index], self.info


    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Steps the simulation using the action tensor and returns the new trajectory.

        Args:
            action (torch.Tensor): Tensor with "action" key that is a tensor with shape (# env, # actions)

        Returns:
            Tuple: Tuple of tensors containing the next state, reward, done, and info dictionary
        """
        next_state, reward, terminated, truncated, self.info = self.env.step(action)

        done = torch.logical_or(terminated, truncated)
        self.state = next_state['policy']

        return next_state['policy'], reward.unsqueeze(-1), done.unsqueeze(-1), self.info
    
    def close(self) -> None:
        """
        Closes the environment and cleans up any resources.
        """
        self.env.close()        


if __name__ == '__main__':
    env = IsaaclabWrapper(env_name="Isaac-Ant-Direct-v0", num_envs=5)
    state, info = env.reset()
    print(state.shape)
    print(state)
    print(info)
    env.close()
