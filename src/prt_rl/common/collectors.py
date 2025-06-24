import torch
from typing import Dict, Optional, List
from prt_rl.env.interface import EnvironmentInterface, EnvParams, MultiAgentEnvParams
from prt_rl.common.loggers import Logger
from prt_rl.common.policies import ActorCriticPolicy

   
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

        Args:
            state (torch.Tensor): The current state of the environment.
        Returns:
            torch.Tensor: A tensor containing the sampled action.
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
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
                - 'state': The states collected.
                - 'action': The actions taken.
                - 'next_state': The next states after taking the actions.
                - 'reward': The rewards received.
                - 'done': The done flags indicating if the episode has ended.
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
            dones.append(done.squeeze(0))

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
    
class ParallelCollector:
    """
    The Parallel Collector collects experience from multiple environments in parallel.
    It resets the environments when the previous experience is done.

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

        Args:
            state (torch.Tensor): The current state of the environment.
        Returns:
            torch.Tensor: A tensor containing the sampled action.
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
            Dict[str, torch.Tensor]: A dictionary containing the collected experience with keys:
                - 'state': The states collected.
                - 'action': The actions taken.
                - 'next_state': The next states after taking the actions.
                - 'reward': The rewards received.
                - 'done': The done flags indicating if the episode has ended.
        """
        # Get the number of steps to take per environment to get at least `num_steps`
        # A trick for ceiling division: (a + b - 1) // b
        num_envs = self.env.get_num_envs()
        num_steps_per_env = (num_steps + num_envs - 1) // num_envs

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        value_estimates = []
        log_probs = []

        for _ in range(num_steps_per_env):
            # Reset the environment if no previous state
            if self.previous_experience is None:
                state, _ = self.env.reset()
            else:
                # Only reset the environments that are done
                state = self.previous_experience["next_state"]
                for i in range(self.previous_experience["done"].shape[0]):
                    if self.previous_experience["done"][i]:
                        # Reset the environment for this index
                        reset_state, _ = self.env.reset_index(i)
                        # Update the previous experience for this index
                        state[i] = reset_state


            # Use random or given policy
            if policy is None:
                action = self._random_action(state)
            elif isinstance(policy, ActorCriticPolicy):
                action, value_est, log_prob = policy.predict(state)
                value_estimates.append(value_est)
                log_probs.append(log_prob)
            else:
                action = policy(state)

            next_state, reward, done, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

            self.previous_experience = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }

        experience = {
            "state": torch.cat(states, dim=0),
            "action": torch.cat(actions, dim=0),
            "next_state": torch.cat(next_states, dim=0),
            "reward": torch.cat(rewards, dim=0),
            "done": torch.cat(dones, dim=0),
        }

        if isinstance(policy, ActorCriticPolicy):
            # Compute the last value estimate
            _, last_value_estimate, _ = self.policy.predict(self.previous_experience['next_state'])
            experience['value_est'] = torch.cat(value_estimates, dim=0)
            experience['log_prob'] = torch.cat(log_probs, dim=0)
            experience['last_value_est'] = last_value_estimate
        
        return experience
