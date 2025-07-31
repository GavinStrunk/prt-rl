import torch
import torch.nn.functional as F
from typing import Optional, List
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils

from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.buffers import ReplayBuffer

class TD3(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)

    """
    def __init__(self,
                 env_params: EnvParams,
                 policy = None,
                 batch_size: int = 256,
                 min_num_steps: int = 1000,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 device: str = 'cpu',
                 ) -> None:
        super().__init__(policy=policy)
        self.env_params = env_params
        self.batch_size = batch_size
        self.min_num_steps = min_num_steps
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.device = torch.device(device)
        self.policy.to(self.device)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform an action based on the current state.

        Args:
            state: The current state of the environment.

        Returns:
            The action to be taken.
        """
        with torch.no_grad():
            return self.policy(state)
        
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              logging_freq: int = 1000,
              evaluator: Evaluator = Evaluator(),
              eval_freq: int = 1000,
              ) -> None:
        """
        Update the agent's knowledge based on the action taken and the received reward.
        This method should implement the TD3 training loop.

        Args:
            env: The environment to interact with.
            total_steps: Total number of steps to train the agent.
            schedulers: Optional list of parameter schedulers.
            logger: Optional logger for logging training progress.
            logging_freq: Frequency of logging.
            evaluator: Evaluator for evaluating the agent's performance.
            eval_freq: Frequency of evaluation.
        """
        logger = logger or Logger.create('blank')
        progress_bar = ProgressBar(total_steps=total_steps)
        num_steps = 0

        # Make collector and do not flatten the experience so the shape is (N, T, ...)
        collector = ParallelCollector(env=env, logger=logger, logging_freq=logging_freq, flatten=False)
        replay_buffer = ReplayBuffer(capacity=100000, device=self.device)

        while num_steps < total_steps:
            # Update Schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)

            # Collect experience dictionary with shape (T, N, ...)
            experience = collector.collect_experience(policy=self.policy, num_steps=self.batch_size)
            num_steps += experience['state'].shape[0] * experience['state'].shape[1]

            # Store experience in replay buffer
            replay_buffer.add(experience)

            # Collect a minimum number of steps in the replay buffer before training
            if replay_buffer.get_size() < self.min_num_steps:
                progress_bar.update(current_step=num_steps, desc="Collecting initial experience...")
                continue
            