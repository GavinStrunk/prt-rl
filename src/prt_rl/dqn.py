import copy
import numpy as np
import torch
from typing import Callable, Optional, List, Dict, Tuple
from prt_rl.env.interface import EnvParams, EnvironmentInterface
from prt_rl.common.replay_buffers import ReplayBuffer, BaseReplayBuffer, PrioritizedReplayBuffer
from prt_rl.common.networks import NatureCNN
from prt_rl.common.decision_functions import EpsilonGreedy
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.collectors import SequentialCollector
# from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.loggers import Logger


class BaseDQN:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, type_: str, **kwargs):
        if type_ not in cls._registry:
            raise ValueError(f"Unknown DQN type: {type_}")
        return cls._registry[type_](**kwargs)
    
    def predict(self,
                 state: torch.Tensor,
                 ) -> torch.Tensor:
        """
        Predict the action using the policy network.

        Args:
            state (torch.Tensor): Current state of the environment.

        Returns:
            torch.Tensor: Action to be taken.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def train(self,
              env: EnvironmentInterface,
              total_frames: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              logging_freq: int = 1000,
              ) -> None:
        """
        Train the DQN agent.
        Args:
            env (EnvironmentInterface): The environment to train on.
            total_frames (int): Total number of frames to train the agent.
            schedulers (List[ParameterScheduler], optional): List of schedulers to update during training. Defaults to None.
            logger (Logger, optional): Logger to log training metrics. Defaults to None.
            logging_freq (int, optional): Frequency of logging. Defaults to 1000.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

@BaseDQN.register("dqn")
class DQN(BaseDQN):
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
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 buffer_size: int = 1_000_000,
                 min_buffer_size: int = 10_000,
                 mini_batch_size: int = 32,
                 max_grad_norm: Optional[float] = 10.0,
                 target_update_freq: Optional[int] = None,
                 polyak_tau: Optional[float] = None,
                 decision_function: Optional[EpsilonGreedy] = None,
                 replay_buffer: Optional[BaseReplayBuffer] = None,
                 dueling: bool = False,
                 device: str = "cuda",
                 ) -> None:
        self.env_params = env_params
        self.decision_function = EpsilonGreedy(epsilon=0.1) if decision_function is None else decision_function
        self.alpha = alpha
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.target_update_freq = target_update_freq
        self.polyak_tau = polyak_tau
        self.device = torch.device(device)
        self._reset_env = True

        # Set either the hard update frequency or the soft update tau
        if (self.target_update_freq is None) == (self.polyak_tau is None):
            raise ValueError("Either target_update_freq or polyak_tau must be set, but not both.")
        
        # Initialize replay buffer
        self.replay_buffer = replay_buffer or ReplayBuffer(capacity=self.buffer_size, device=torch.device(device))

        # Initialize Policy
        self.policy = NatureCNN(
            state_shape=self.env_params.observation_shape, 
            action_len=self.env_params.action_max+1,
            dueling=dueling
            ).to(self.device)

        # Initialize target network
        self.target = copy.deepcopy(self.policy).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            params=self.policy.parameters(),
            lr=self.alpha
        )

    def _compute_td_targets(self, 
                            next_state: torch.Tensor, 
                            reward: torch.Tensor, 
                            done: torch.Tensor
                            ) -> torch.Tensor:
        """
        Compute the TD target values for the sampled batch.

        """
        target_values = self.target(next_state)
        td_target = reward + (1-done.float()) * self.gamma * torch.max(target_values, dim=1, keepdim=True)[0]
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
        return loss, td_error
    
    @staticmethod
    def _polyak_update(policy: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
        """
        Polyak update for the target network.

        .. math::
            \Theta_{target} = \tau * \Theta_{\pi} + (1 - \tau) * \Theta_{target}

        References:
        [1] https://github.com/DLR-RM/stable-baselines3/issues/93
        """
        for target_params, policy_params in zip(target.parameters(), policy.parameters()):
            # Update target network parameters using Polyak averaging
            params = tau * policy_params.data + (1 - tau) * target_params.data
            target_params.data.copy_(params)

    def _hard_update(self, policy: torch.nn.Module, target: torch.nn.Module) -> None:
        """
        Hard update for the target network.

        .. math::
            \Theta_{target} = \Theta_{\pi}

        """
        for target_params, policy_params in zip(target.parameters(), policy.parameters()):
            # Update target network parameters with policy network parameters
            target_params.data.copy_(policy_params.data)

    # @staticmethod
    # def _collect_experience(env: EnvironmentInterface, policy: Callable, last_experience: dict) -> Tuple[Dict[str, torch.Tensor], dict]:
    #     """
    #     Collect experience from a single step of the environment.
    #     Args:
    #         env (EnvironmentInterface): The environment from which to collect data.
    #     """
    #     if last_experience == {} or last_experience["done"]:
    #         state, _ = env.reset()
    #     else:
    #         state = last_experience["next_state"]
        
    #     action = policy(state)
    #     next_state, reward, done, info = env.step(action)

    #     return {
    #         "state": state,
    #         "action": action,
    #         "next_state": next_state,
    #         "reward": reward,
    #         "done": done
    #     }, info


    def predict(self,
                 state: torch.Tensor,
                 ) -> torch.Tensor:
        """
        Predict the action using the policy network.

        Args:
            state (torch.Tensor): Current state of the environment.

        Returns:
            torch.Tensor: Action to be taken.
        """
        q_val = self.policy(state)
        action = self.decision_function.select_action(q_val)
        return action


    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              logging_freq: int = 1,
              train_freq: int = 4,
              gradient_steps: int = 1,
              ) -> None:
        """
        Train the DQN agent.
        Args:
            env (EnvironmentInterface): The environment to train on.
            total_steps (int): Total number of steps to train the agent.
            schedulers (List[ParameterScheduler], optional): List of schedulers to update during training. Defaults to None.
            logger (Logger, optional): Logger to log training metrics. Defaults to None.
            logging_freq (int, optional): Frequency of logging. Defaults to 1000.
        """
        logger = logger or Logger.create('blank')
        collector = SequentialCollector(env, logger=logger, logging_freq=logging_freq)
        experience = {}
        td_errors = []
        losses = []
        iteration_count = 0
        num_steps = 0

        # Collect initial random experience to fill the replay buffer
        # @todo update with RandomAgent
        random_experience = collector.collect_experience(policy=self.predict, num_steps=self.min_buffer_size)
        num_steps += len(random_experience)
        self.replay_buffer.add(random_experience)

        # Run DQN training loop
        while num_steps < total_steps:
            # Collect experience and add to replay buffer
            experience = collector.collect_experience(policy=self.predict, num_steps=1)
            num_steps += len(experience)
            self.replay_buffer.add(experience)

            # Only train at a rate of the training frequency
            iteration_count += 1
            if iteration_count % train_freq == 0:
                # If minimum number of samples in replay buffer, sample a batch
                batch_data = self.replay_buffer.sample(batch_size=self.mini_batch_size)

                # Compute TD Target Values
                td_targets = self._compute_td_targets(
                    next_state=batch_data["next_state"],
                    reward=batch_data["reward"],
                    done=batch_data["done"]
                )

                # Compute Q values 
                q = self.policy(batch_data["state"])
                qsa = torch.gather(q, dim=1, index=batch_data["action"].to(torch.int64))

                # Compute loss
                loss, td_error = self._compute_loss(
                    td_target=td_targets,
                    qsa=qsa,
                )
                td_errors.append(td_error.abs().mean().item())
                losses.append(loss.mean().item())

                # Optimize policy model parameters
                self.optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm is not None:
                    # Clip gradients if max_grad_norm is specified
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        max_norm=self.max_grad_norm
                    )
                
                self.optimizer.step()

                # Update sample priorities if this is a prioritized replay buffer
                if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                    self.replay_buffer.update_priorities(batch_data["indices"], td_error)

                training_steps += 1

                # Update target network with either hard or soft update
                if self.target_update_freq is not None and training_steps % self.target_update_freq == 0:
                    self._hard_update(self.policy, self.target)
                elif self.polyak_tau is not None:
                    # Polyak update
                    self._polyak_update(self.policy, self.target, self.polyak_tau)
                
                # iteration_count = 0

            # Log training metrics
            if iteration_count % logging_freq == 0:
                if schedulers is not None:
                    for scheduler in schedulers:
                        logger.log_scalar(name=scheduler.parameter_name, value=getattr(scheduler.obj, scheduler.parameter_name), iteration=num_frames)
                logger.log_scalar(name="td_error", value=td_errors[-1], iteration=num_steps)
                logger.log_scalar(name="loss", value=losses[-1], iteration=num_steps)
            
            # Update schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=iteration_count)

        # Clean up for saving the agent
        # Clear the replay buffer because it can be large
        self.replay_buffer.clear()

@BaseDQN.register("double_dqn")
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
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 buffer_size: int = 1_000_000,
                 min_buffer_size: int = 10_000,
                 mini_batch_size: int = 32,
                 max_grad_norm: Optional[float] = None,
                 target_update_freq: Optional[int] = None,
                 polyak_tau: Optional[float] = None,
                 decision_function: Optional[EpsilonGreedy] = None,
                 replay_buffer: Optional[BaseReplayBuffer] = None,
                 dueling: bool = False,
                 device: str = "cuda",
                 ) -> None:
        super().__init__(
            env_params=env_params,
            alpha=alpha,
            gamma=gamma,
            buffer_size=buffer_size,
            min_buffer_size=min_buffer_size,
            mini_batch_size=mini_batch_size,
            max_grad_norm=max_grad_norm,
            target_update_freq=target_update_freq,
            polyak_tau=polyak_tau,
            decision_function=decision_function,
            replay_buffer=replay_buffer,
            dueling=dueling,
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
        action_selections = self.policy(next_state).argmax(dim=1, keepdim=True)
        td_target = reward + (1-done.float()) * self.gamma * torch.gather(self.target(next_state), dim=1, index=action_selections)
        return td_target