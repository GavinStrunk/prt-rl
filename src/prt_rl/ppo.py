import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from typing import Optional, List, Any
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.utils.policy import ActorCriticPolicy
from prt_rl.utils.loggers import Logger
from prt_rl.utils.schedulers import ParameterScheduler
from prt_rl.utils.progress_bar import ProgressBar
from prt_rl.utils.metrics import MetricTracker

class PPO:
    def __init__(self,
                 env: EnvironmentInterface,
                 policy: Optional[ActorCriticPolicy] = None,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 schedulers: Optional[List[ParameterScheduler]] = None,
                 progress_bar: Optional[ProgressBar] = ProgressBar,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 learning_rate: float = 3e-4,
                 num_optim_steps: int = 10,
                 mini_batch_size: int = 32,
                 ) -> None:
        self.env = env
        self.policy = policy or ActorCriticPolicy(self.env.get_parameters())
        self.logger = logger or Logger()
        self.metric_tracker = metric_tracker or MetricTracker()
        self.schedulers = schedulers or []
        self.progress_bar = progress_bar
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_optim_steps = num_optim_steps
        self.mini_batch_size = mini_batch_size

        self.actor_optimizer = torch.optim.Adam(self.policy.get_actor_network().parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.policy.get_critic_network().parameters(), lr=learning_rate)

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        """
        Sets a name value parameter

        Args:
            name (str): The name of the parameter
            value (Any): The value of the parameter

        Raises:
            ValueError: If the parameter is not found
        """
        try:
            self.policy.set_parameter(name, value)
        except ValueError:
            raise ValueError(f"Parameter {name} not found in TDTrainer")

    def compute_returns(self, rewards, dones):
        """
        Compute rewards-to-go for each timestep in an episode.
        """
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

    def compute_advantages(self, returns, values):
        return (returns - values)

    def compute_actor_loss(self, new_log_probs, old_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_adv = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        return -torch.min(ratio * advantages, clipped_adv).mean()

    def compute_critic_loss(self, new_values, returns):
        return F.mse_loss(new_values, returns)

    def train(self,
              num_episodes: int
              ) -> None:

        # Initialize progress bar
        if self.progress_bar is not None:
            self.progress_bar = self.progress_bar(total_frames=num_episodes, frames_per_batch=1)

        cumulative_reward = 0
        # Initialize metrics
        for i in range(num_episodes):

            # Step schedulers if there are any
            for sch in self.schedulers:
                name = sch.parameter_name
                new_val = sch.update(i)
                self.set_parameter(name, new_val)
                self.logger.log_scalar(name, new_val, iteration=i)

            # Collect 1 trajectory
            experience_buffer = []
            value_estimates = []
            action_log_probs = []
            obs_td = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action_td = self.policy.get_action(obs_td)
                value_estimates.append(self.policy.get_value_estimates())
                action_log_probs.append(self.policy.get_log_probs(action_td['action']))

                obs_td = self.env.step(action_td)

                experience_buffer.append(obs_td.copy())

                episode_reward += obs_td['next', 'reward']
                done = obs_td['next', 'done']

                obs_td = self.env.step_mdp(obs_td)

            # Stack experiences
            experience_buffer = torch.cat(experience_buffer, dim=0)
            action_log_probs = torch.cat(action_log_probs, dim=0).detach()
            value_estimates = torch.cat(value_estimates, dim=0)

            # Compute Returns - Rewards to go
            returns = self.compute_returns(rewards=experience_buffer['next', 'reward'], dones=experience_buffer['next', 'done'])
            returns = returns.detach()

            # Compute Advantages
            advantages = self.compute_advantages(returns, value_estimates)
            advantages = advantages.detach()

            # Learn
            for _ in range(self.num_optim_steps):
                for i in range(0, len(experience_buffer), self.mini_batch_size):
                    # Get batch data
                    batch_experience = experience_buffer[i:i + self.mini_batch_size]
                    batch_actions = batch_experience['action'].clone()
                    batch_returns = returns[i:i + self.mini_batch_size]
                    batch_advantages = advantages[i:i + self.mini_batch_size]
                    batch_log_probs = action_log_probs[i:i + self.mini_batch_size]

                    # Recompute log probs and value estimates
                    _ = self.policy.get_action(batch_experience)
                    new_values = self.policy.get_value_estimates()
                    new_action_log_probs = self.policy.get_log_probs(batch_actions)

                    # Compute Actor (Policy) Loss
                    actor_loss = self.compute_actor_loss(
                        new_log_probs=new_action_log_probs,
                        old_log_probs=batch_log_probs,
                        advantages=batch_advantages
                    )

                    # Compute Critic Loss
                    critic_loss = self.compute_critic_loss(new_values=new_values, returns=batch_returns)

                    # Optimization Steps
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

            cumulative_reward += episode_reward
            self.progress_bar.update(episode_reward, cumulative_reward)
            self.logger.log_scalar('episode_reward', episode_reward, iteration=i)
            self.logger.log_scalar('cumulative_reward', cumulative_reward, iteration=i)

            if episode_reward >= 500:
                print(f"Won the game: {episode_reward}")