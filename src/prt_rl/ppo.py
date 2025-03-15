"""
Proximal Policy Optimization (PPO)

Reference:
[1] https://arxiv.org/abs/1707.06347
"""
import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from typing import Optional, List, Any
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.policy import ActorCriticPolicy
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.metrics import MetricTracker
from prt_rl.common.trainers import ActorCriticTrainer

class PPO(ActorCriticTrainer):
    """
    Proximal Policy Optimization (PPO)

    """
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
        policy = policy or ActorCriticPolicy(env.get_parameters())

        super().__init__(env=env, policy=policy, logger=logger, metric_tracker=metric_tracker, schedulers=schedulers,
                         progress_bar=progress_bar, num_optimization_steps=num_optim_steps,
                         mini_batch_size=mini_batch_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_optim_steps = num_optim_steps
        self.mini_batch_size = mini_batch_size


    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.policy.get_actor_network().parameters(), lr=self.learning_rate),
            torch.optim.Adam(self.policy.get_critic_network().parameters(), lr=self.learning_rate)
        ]
        return optimizers

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

    def compute_loss(self,
                     batch_experience: TensorDict,
                     batch_returns: torch.Tensor,
                     batch_advantages: torch.Tensor,
                     batch_action_log_probs: torch.Tensor,
                     ) -> List:
        batch_actions = batch_experience['action'].clone()

        # Recompute log probs and value estimates
        _ = self.policy.get_action(batch_experience)
        new_values = self.policy.get_value_estimates()
        new_action_log_probs = self.policy.get_log_probs(batch_actions)

        ratio = torch.exp(new_action_log_probs - batch_action_log_probs)
        clipped_adv = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
        clip_loss = -torch.min(ratio * batch_advantages, clipped_adv).mean()

        value_loss = F.mse_loss(new_values, batch_returns)

        return [clip_loss, value_loss]
