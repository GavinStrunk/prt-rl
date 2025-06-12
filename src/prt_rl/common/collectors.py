import torch
from typing import Dict, Optional, List
from prt_rl.env.interface import EnvironmentInterface, EnvParams, MultiAgentEnvParams
from prt_rl.common.loggers import Logger

class Collector:
    def __init__(self, 
                 env, 
                 ):
        self.env = env

        # Detect the number of environments
        state, _ = self.env.reset()
        self.num_envs = state.shape[0]

    def collect(self, 
                policy: None, 
                num_steps: int
                ) -> Dict[str, torch.Tensor]:
        
        steps_per_env = num_steps // self.num_envs

        # Reset all environments
        state, _ = self.env.reset()

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        for _ in range(steps_per_env):
            with torch.no_grad():
                action = policy(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = torch.logical_or(terminated, truncated)

            # Append data
            states.append(state)
            actions.append(action)
            next_states.append(torch.from_numpy(next_state).float())
            rewards.append(torch.from_numpy(reward).float())
            dones.append(torch.from_numpy(done).bool())

            # Reset only where done
            if torch.any(done):
                next_state[done] = self.env.reset(seed=None)[0][done]


        return {
            "state": torch.cat(states),
            "action": torch.cat(actions),
            "next_state": torch.cat(next_states),
            "reward": torch.cat(rewards),
            "done": torch.cat(dones),
        }
    
class SequentialCollector:
    """
    The Sequential Collector collects experience from a single environment sequentially.
    It resets the environment when the previous experience is done.

    Args:
        env (EnvironmentInterface): The environment to collect experience from.
        logger (Optional[Logger]): Optional logger for logging information. Defaults to a new Logger instance.
        logging_freq (int): Frequency of logging experience collection. Defaults to 1.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 logger: Optional[Logger] = None,
                 logging_freq: int = 1
                 ) -> None:
        self.env = env
        self.env_params = env.get_parameters()
        self.logger = logger or Logger.create('blank')
        self.logging_freq = logging_freq
        self.previous_experience = None
        self.collected_steps = 0
        self.previous_episode_reward = 0
        self.previous_episode_length = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.cumulative_reward = 0
        self.num_episodes = 0

    def _random_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Randomly samples an action from action space.

        Returns:
            TensorDict: Tensordict with the "action" key added
        """
        if isinstance(self.env_params, EnvParams):
            ashape = (state.shape[0], self.env_params.action_len)
            params = self.env_params
        elif isinstance(self.env_params, MultiAgentEnvParams):
            ashape = (state.shape[0], self.env_params.num_agents, self.env_params.agent.action_len)
            params = self.env_params.agent
        else:
            raise ValueError("env_params must be a EnvParams or MultiAgentEnvParams")

        if not params.action_continuous:
            # Add 1 to the high value because randint samples between low and 1 less than the high: [low,high)
            action = torch.randint(low=params.action_min, high=params.action_max + 1,
                                   size=ashape)
        else:
            action = torch.rand(size=ashape)

            # Scale the random [0,1] actions to the action space [min,max]
            max_actions = torch.tensor(params.action_max).unsqueeze(0)
            min_actions = torch.tensor(params.action_min).unsqueeze(0)
            action = action * (max_actions - min_actions) + min_actions

        return action 

    def collect_experience(self,
                           policy = None,
                           num_steps: int = 1
                           ) -> Dict[str, torch.Tensor]:
        """
        Collects the given number of experiences from the environment using the provided policy.
        Args:
            policy (callable): A callable that takes a state and returns an action.
            num_steps (int): The number of steps to collect experience for. Defaults to 1.
        Returns:
            List[dict]: A list of experience dictionaries, each containing state, action, next_state, reward, and done.
        """
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        for _ in range(num_steps):
            # Reset the environment if no previous state
            if self.previous_experience is None or self.previous_experience["done"]:
                state, _ = self.env.reset()
            else:
                state = self.previous_experience["next_state"]

            # Use random or given policy
            action = self._random_action(state) if policy is None else policy(state)

            next_state, reward, done, _ = self.env.step(action)

            states.append(state.squeeze(0))
            actions.append(action.squeeze(0))
            next_states.append(next_state.squeeze(0))
            rewards.append(reward.squeeze(0))
            dones.append(torch.tensor([done], dtype=torch.bool, device=reward.device))

            self.collected_steps += 1
            self.current_episode_reward += reward.sum().item()
            self.current_episode_length += 1
            self.cumulative_reward += reward.sum().item()

            if done:
                self.previous_episode_reward = self.current_episode_reward
                self.previous_episode_length = self.current_episode_length
                self.current_episode_reward = 0
                self.current_episode_length = 0
                self.num_episodes += 1

            self.previous_experience = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }

        if self.collected_steps % self.logging_freq == 0:
            self.logger.log_scalar(name='episode_reward', value=self.previous_episode_reward, iteration=self.collected_steps)
            self.logger.log_scalar(name='episode_length', value=self.previous_episode_length, iteration=self.collected_steps)
            self.logger.log_scalar(name='cumulative_reward', value=self.cumulative_reward, iteration=self.collected_steps)
            self.logger.log_scalar(name='episode_number', value=self.num_episodes, iteration=self.collected_steps)

        return {
            "state": torch.stack(states, dim=0),
            "action": torch.stack(actions, dim=0),
            "next_state": torch.stack(next_states, dim=0),
            "reward": torch.stack(rewards, dim=0),
            "done": torch.stack(dones, dim=0),
        }
    