import numpy as np
import torch
from typing import Optional, List
from prt_rl.agent import BaseAgent
from prt_rl.env.interface import EnvironmentInterface, EnvParams
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.loggers import Logger
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator

from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.collectors import SequentialCollector
from prt_rl.common.policies import DistributionPolicy
from prt_rl.common.distributions import Categorical, Normal


class DAgger(BaseAgent):
    """
    Dataset Aggregation from Demonstrations (DAgger) agent.

    Args:
        env_params (EnvParams): Environment parameters.
        policy (Optional[DistributionPolicy]): The policy to be used by the agent. If None, a default policy will be created based on the environment parameters.
        buffer_size (int): Size of the replay buffer. Default is 10000.
        learning_rate (float): Learning rate for the optimizer. Default is 1e-3.
        optim_steps (int): Number of optimization steps per training iteration. Default is 1.
        mini_batch_size (int): Size of the mini-batch for training. Default is 32.
        max_grad_norm (float): Maximum gradient norm for gradient clipping. Default is 10.0.
        device (str): Device to run the agent on (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
    """
    def __init__(self,
                 env_params: EnvParams, 
                 policy: Optional[DistributionPolicy] = None,
                 buffer_size: int = 10000,
                 learning_rate: float = 1e-3,
                 optim_steps: int = 1,
                 mini_batch_size: int = 32,
                 max_grad_norm: float = 10.0,
                 device: str = 'cpu',
                 ) -> None:
        self.env_params = env_params
        self.policy = policy if policy is not None else DistributionPolicy(env_params=env_params)
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.optim_steps = optim_steps
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.loss_function = self._get_loss_function(self.policy)

    @staticmethod
    def _get_loss_function(policy: DistributionPolicy) -> torch.nn.Module:
        """
        Returns the loss function used for training the policy based on the type of distribution.

        Args:
            policy (DistributionPolicy): The policy for which to get the loss function.
        Returns:
            torch.nn.Module: The loss function to be used for training.
        Raises:
            ValueError: If the distribution type is not supported.
        """
        if issubclass(policy.distribution, Categorical):
            # For categorical distributions, use CrossEntropyLoss
            return torch.nn.CrossEntropyLoss()
        elif issubclass(policy.distribution, Normal):
            # For continuous distributions, use MSELoss
            return torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported distribution type {policy.distribution.__class__} loss function.")
    
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform an action based on the current state using the policy.
        
        Args:
            state: The current state of the environment.
        
        Returns:
            The action to be taken by the policy.
        """
        with torch.no_grad():
            return self.policy(state)
    
    def train(self,
              env: EnvironmentInterface,
              expert_policy: BaseAgent,
              experience_buffer: ReplayBuffer,
              total_steps: int,
              schedulers: List[ParameterScheduler] = [],              
              logger: Optional[Logger] = None,
              logging_freq: int = 1,              
              evaluator: Evaluator = Evaluator(),
              eval_freq: int = 1000,
              ) -> None:
        """
        Train the DAgger agent using the provided environment and expert policy.

        Args:
            env (EnvironmentInterface): The environment in which the agent will operate.
            expert_policy (BaseAgent): The expert agent to provide actions for the states.
            experience_buffer (ReplayBuffer): The replay buffer to store experiences.
            total_steps (int): Total number of training steps to perform.
            schedulers (List[ParameterScheduler]): List of parameter schedulers to update during training.
            logger (Optional[Logger]): Logger for logging training progress. If None, a default logger will be created.
            logging_freq (int): Frequency of logging training progress.
            evaluator (Evaluator): Evaluator to evaluate the agent periodically.
            eval_freq (int): Frequency of evaluation during training.
        """
        logger = logger or Logger.create('blank')
        progress_bar = ProgressBar(total_steps=total_steps)

        # Resize the replay buffer with size: initial experience + total_steps
        experience_buffer.resize(new_capacity=experience_buffer.size + self.buffer_size)

        # Add initial experience to the replay buffer
        collector = SequentialCollector(env=env, logger=logger, logging_freq=logging_freq)

        num_steps = 0

        while num_steps < total_steps:
            # Update schedulers if any
            for scheduler in schedulers:
                scheduler.update(current_step=num_steps)

            # Collect experience using the current policy
            policy_experience = collector.collect_experience(policy=self.policy, num_steps=1000)
            num_steps += policy_experience['state'].shape[0]

            # Get expert action for each state in the collected experience
            expert_actions = expert_policy(policy_experience['state'])

            # Update the policy experience with expert actions
            policy_experience['action'] = expert_actions

            # Add the policy experience to the replay buffer
            experience_buffer.add(policy_experience)

            # Optimize the policy
            losses = []
            for _ in range(self.optim_steps):
                for batch in experience_buffer.get_batches(batch_size=self.mini_batch_size):
                    # Compute the loss between the policy's actions and the expert's actions
                    if not self.env_params.action_continuous:
                        policy_logits = self.policy.get_logits(batch['state'])
                        loss = self.loss_function(policy_logits, batch['action'].squeeze(1))
                    else:
                        policy_actions = self.policy(batch['state'])
                        loss = self.loss_function(policy_actions, batch['action'])

                    losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()
            
            progress_bar.update(num_steps, desc=f"Episode Reward: {collector.previous_episode_reward:.2f}, "
                                                                   f"Episode Length: {collector.previous_episode_length}, "
                                                                   f"Loss: {np.mean(losses):.4f},")

            # Log the training progress
            if num_steps % logging_freq == 0:
                for scheduler in schedulers:
                    logger.log_scalar(name=scheduler.parameter_name, value=getattr(scheduler.obj, scheduler.parameter_name), iteration=num_steps)
                logger.log_scalar(name='loss', value=np.mean(losses), iteration=num_steps)

            # Evaluate the agent periodically
            if num_steps % eval_freq == 0:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)

        evaluator.close()
