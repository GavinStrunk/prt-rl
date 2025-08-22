"""
Deep Q-Network (DQN) Agents
"""
import copy
from dataclasses import dataclass
import numpy as np
import torch
from typing import Optional, List, Tuple
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.buffers import ReplayBuffer, BaseBuffer, PrioritizedReplayBuffer
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.collectors import ParallelCollector
from prt_rl.common.loggers import Logger
from prt_rl.agent import BaseAgent
from prt_rl.common.policies import QValuePolicy
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
import prt_rl.common.utils as utils


@dataclass
class DQNConfig:
    """
    Hyperparameters for the DQN agent.

    Args:
        buffer_size (int): Size of the replay buffer. Default is 1_000_000.
        min_buffer_size (int): Minimum size of the replay buffer before training. Default is 10_000.
        mini_batch_size (int): Size of the mini-batch for training. Default is 32.
        learning_rate (float): Learning rate for the optimizer. Default is 0.1.
        gamma (float): Discount factor for future rewards. Default is 0.99.
        max_grad_norm (float): Maximum gradient norm for gradient clipping. Default is None.
        target_update_freq (int): Frequency of target network updates. Default is 1.
        polyak_tau (float): Polyak averaging coefficient for target network updates. Default is None.
        train_freq (int): Frequency of training steps. Default is 1.
        gradient_steps (int): Number of gradient steps per training iteration. Default is 1.
    """
    buffer_size: int = 1_000_000
    min_buffer_size: int = 10_000
    mini_batch_size: int = 32
    learning_rate: float = 0.1
    gamma: float = 0.99
    max_grad_norm: Optional[float] = None
    target_update_freq: int = 1
    polyak_tau: Optional[float] = None
    train_freq: int = 1
    gradient_steps: int = 1

class DQN(BaseAgent):
    """
    Deep Q-Network (DQN) agent for reinforcement learning.

    Args:
        env_params (EnvParams): Environment parameters.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        buffer_size (int, optional): Size of the replay buffer. Defaults to 1_000_000.
        min_buffer_size (int, optional): Minimum size of the replay buffer before training. Defaults to 10_000.
        mini_batch_size (int, optional): Size of the mini-batch for training. Defaults to 32.
        max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to None.
        target_update_freq (int, optional): Frequency of target network updates. Defaults to None.
        polyak_tau (float, optional): Polyak averaging coefficient for target network updates. Defaults to None.
        decision_function (EpsilonGreedy, optional): Decision function for action selection. Defaults to EpsilonGreedy(epsilon=0.1).
        replay_buffer (BaseReplayBuffer, optional): Replay buffer for storing experiences. Defaults to None.
        device (str, optional): Device for computation ('cpu' or 'cuda'). Defaults to 'cuda'.

    References:
    [1] https://openai.com/index/openai-baselines-dqn/
    [2] https://github.com/openai/baselines/tree/master
    [3] Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
    """
    def __init__(self,
                 env_params: EnvParams,
                 policy: QValuePolicy | None = None,
                 replay_buffer: Optional[BaseBuffer] = None,
                 config: DQNConfig = DQNConfig(),
                #  buffer_size: int = 1_000_000,
                #  min_buffer_size: int = 10_000,
                #  mini_batch_size: int = 32,                 
                #  learning_rate: float = 0.1,
                #  gamma: float = 0.99,
                #  max_grad_norm: Optional[float] = 10.0,
                #  target_update_freq: int = 1,
                #  polyak_tau: Optional[float] = None,
                #  train_freq: int = 1,
                #  gradient_steps: int = 1,
                 device: str = "cpu",
                 ) -> None:
        super().__init__()
        if env_params.action_continuous:
            raise ValueError("DQN only supports discrete action spaces")
        
        self.env_params = env_params
        self.config = config
        # self.learning_rate = learning_rate
        # self.gamma = gamma
        # self.buffer_size = buffer_size
        # self.min_buffer_size = min_buffer_size
        # self.mini_batch_size = mini_batch_size
        # self.max_grad_norm = max_grad_norm
        # self.target_update_freq = target_update_freq
        # self.polyak_tau = polyak_tau
        # self.train_freq = train_freq
        # self.gradient_steps = gradient_steps
        self.device = torch.device(device)
        self._reset_env = True

        self.policy = policy if policy is not None else QValuePolicy(env_params=env_params)
        self.policy.to(self.device)

        # Initialize replay buffer
        self.replay_buffer = replay_buffer or ReplayBuffer(capacity=self.config.buffer_size, device=torch.device(device))

        # Initialize target network
        self.target = copy.deepcopy(self.policy).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            params=self.policy.parameters(),
            lr=self.config.learning_rate
        )

    def _compute_td_targets(self, 
                            next_state: torch.Tensor, 
                            reward: torch.Tensor, 
                            done: torch.Tensor
                            ) -> torch.Tensor:
        """
        Compute the TD target values for the sampled batch.

        """
        target_values = self.target.get_q_values(next_state)
        td_target = reward + (1-done.float()) * self.config.gamma * torch.max(target_values, dim=1, keepdim=True)[0]
        return td_target

    @staticmethod
    def _compute_loss(td_target: torch.Tensor,
                      qsa: torch.Tensor,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the sampled batch.

        """
        td_error = td_target - qsa
        loss = torch.mean(td_error ** 2)
        # loss = torch.nn.functional.smooth_l1_loss(qsa, td_target)
        return loss, td_error

    def predict(self,
                 state: torch.Tensor,
                 deterministic: bool = False
                 ) -> torch.Tensor:
        """
        Predict the action using the policy network.

        Args:
            state (torch.Tensor): Current state of the environment.
            deterministic (bool): If True, the action will be selected deterministically.

        Returns:
            torch.Tensor: Action to be taken.
        """
        with torch.no_grad():
            return self.policy(state, deterministic=deterministic)

    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              evaluator: Evaluator = Evaluator(),
              show_progress: bool = True
              ) -> None:
        """
        Train the DQN agent.
        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of steps to train the agent.
            schedulers (List[ParameterScheduler], optional): List of schedulers to update during training. Defaults to None.
            logger (Logger, optional): Logger to log training metrics. Defaults to None.
            evaluator (Evaluator): Evaluator to evaluate the agent periodically.
            show_progress (bool): If True, show a progress bar during training.            
        """
        logger = logger or Logger.create('blank')

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        # Setup up collector to return experience with shape (B, ...)
        collector = ParallelCollector(env, logger=logger, flatten=True)
        
        experience = {}
        num_steps = 0
        training_steps = 0

        # Run DQN training loop
        while num_steps < total_steps:
            # Update schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)
            
            # Collect initial experience until the replay buffer is filled
            if self.replay_buffer.get_size() < self.config.min_buffer_size:
                experience = collector.collect_experience(num_steps=1)
                num_steps += experience["state"].shape[0]
                self.replay_buffer.add(experience)

                if show_progress:
                    progress_bar.update(current_step=num_steps, desc="Collecting initial experience...")
                continue
            
            # Collect experience and add to replay buffer
            self.policy.eval()
            experience = collector.collect_experience(policy=self.policy, num_steps=1)
            num_steps += experience["state"].shape[0]
            self.replay_buffer.add(experience)

            # Only train at a rate of the training frequency
            td_errors = []
            losses = []
            if num_steps % self.config.train_freq == 0:
                self.policy.train()

                for _ in range(self.config.gradient_steps):
                    # If minimum number of samples in replay buffer, sample a batch
                    batch_data = self.replay_buffer.sample(batch_size=self.config.mini_batch_size)

                    # Compute TD Target Values
                    with torch.no_grad():
                        td_targets = self._compute_td_targets(
                            next_state=batch_data["next_state"],
                            reward=batch_data["reward"],
                            done=batch_data["done"]
                        )

                    # Compute Q values 
                    q = self.policy.get_q_values(batch_data["state"])
                    qsa = torch.gather(q, dim=1, index=batch_data["action"].long())

                    # Compute loss
                    loss, td_error = self._compute_loss(
                        td_target=td_targets,
                        qsa=qsa,
                    )
                    td_errors.append(td_error.abs().mean().item())
                    losses.append(loss.item())

                    # Optimize policy model parameters
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.max_grad_norm)
                    self.optimizer.step()

                    # Update sample priorities if this is a prioritized replay buffer
                    if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                        self.replay_buffer.update_priorities(batch_data["indices"], td_error)

                training_steps += 1

                if show_progress:
                    tracker = collector.get_metric_tracker()
                    progress_bar.update(current_step=num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}, "
                                                                   f"Episode Length: {tracker.last_episode_length}, "
                                                                   f"Loss: {np.mean(losses):.4f},")

            # Update target network with either hard or soft update
            if num_steps % self.config.target_update_freq == 0:
                if self.config.polyak_tau is None:
                    utils.hard_update(target=self.target, network=self.policy)
                else:
                    # Polyak update
                    utils.polyak_update(target=self.target, network=self.policy, tau=self.config.polyak_tau)
                
            # Log training metrics
            if logger.should_log(num_steps):
                if schedulers is not None:
                    for scheduler in schedulers:
                        logger.log_scalar(name=scheduler.parameter_name, value=getattr(scheduler.obj, scheduler.parameter_name), iteration=num_steps)
                logger.log_scalar(name="td_error", value=np.mean(td_errors), iteration=num_steps)
                logger.log_scalar(name="loss", value=np.mean(losses), iteration=num_steps)

            if evaluator is not None:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)
            
        if evaluator is not None:
            evaluator.close()

        # Clean up for saving the agent
        # Clear the replay buffer because it can be large
        self.replay_buffer.clear()

class DoubleDQN(DQN):
    """
    Double DQN agent for reinforcement learning.

    Args:
        env_params (EnvParams): Environment parameters.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        buffer_size (int, optional): Size of the replay buffer. Defaults to 1_000_000.
        min_buffer_size (int, optional): Minimum size of the replay buffer before training. Defaults to 10_000.
        mini_batch_size (int, optional): Size of the mini-batch for training. Defaults to 32.
        max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to None.
        target_update_freq (int, optional): Frequency of target network updates. Defaults to None.
        polyak_tau (float, optional): Polyak averaging coefficient for target network updates. Defaults to None.
        decision_function (EpsilonGreedy, optional): Decision function for action selection. Defaults to EpsilonGreedy(epsilon=0.1).
        device (str, optional): Device for computation ('cpu' or 'cuda'). Defaults to 'cuda'.
    
    References:
    [1] https://github.com/Curt-Park/rainbow-is-all-you-need
    """
    def __init__(self,
                 env_params: EnvParams,
                 policy: QValuePolicy | None = None,
                 replay_buffer: Optional[BaseBuffer] = None,
                 config: DQNConfig = DQNConfig(),
                 device: str = "cpu",
                 ) -> None:
        super().__init__(
            env_params=env_params,
            policy=policy,
            replay_buffer=replay_buffer,
            config=config,
            device=device
        )
    
    def _compute_td_targets(self, 
                            next_state: torch.Tensor, 
                            reward: torch.Tensor, 
                            done: torch.Tensor
                            ) -> torch.Tensor:
        """
        DDQN separates the parameters used for action selection and action evaluation for the max operation. The policy network is used to select the action, and the target network is used to evaluate the action.

        This is done to reduce the overestimation bias of Q-learning.
        The TD target is computed as follows: 
        .. math::
            Y_t^{DDQN} = R_{t+1} + \gamma Q_{target}(s_{t+1}, \argmax_a Q_{policy}(s_{t+1}, a))
            
            where :math:`Q_{policy}` is the policy network and :math:`Q_{target}` is the target network.

        Args:
            next_state (torch.Tensor): Next state of the environment.
            reward (torch.Tensor): Reward received from the environment.
            done (torch.Tensor): Done flag indicating if the episode has ended.
        Returns:
            torch.Tensor: TD target values.
        """
        action_selections = self.policy.get_q_values(next_state).argmax(dim=1, keepdim=True)
        td_target = reward + (1-done.float()) * self.config.gamma * torch.gather(self.target.get_q_values(next_state), dim=1, index=action_selections)
        return td_target