from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from torch import Tensor
from typing import Optional, List, Tuple, Dict
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.agent import Agent
from prt_rl.common.policies import TabularPolicy
from prt_rl.common.decision_functions import DecisionFunction
from prt_rl.common.loggers import Logger
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.evaluators import Evaluator
from prt_rl.common.collectors import Collector

@dataclass
class QLearningConfig:
    gamma: float = 0.99
    alpha: float = 0.1

class QLearningPolicy(TabularPolicy):   
    """
    Q-Learning Policy implementation using a tabular Q-table.
    
    This policy stores state-action values in a table and uses a decision function
    to select actions. It supports both stochastic action selection (during training)
    and deterministic action selection (during evaluation).
    
    Args:
        qtable: A 2D tensor of shape (num_states, num_actions) containing Q-values
            for each state-action pair.
        decision_function: A function that takes action values and returns an action
            (e.g., epsilon-greedy, softmax).
    
    Raises:
        ValueError: If qtable is not a 2D tensor.
    
    Attributes:
        decision_function: The decision function used for action selection.
        table: The Q-table inherited from TabularPolicy.
    """
    def __init__(self, 
                 qtable: Tensor, 
                 decision_function: DecisionFunction
                 ) -> None:
        if qtable.dim() != 2:
            raise ValueError(f"Q-table must be a 2D tensor of shape (num_states, num_actions), but got shape {qtable.shape}.")
        
        super().__init__(table=qtable, decision_function=decision_function)

    def act(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Select an action based on the observation.
        
        Args:
            obs: The current state/observation as a tensor. (B, S)
            deterministic: If True, selects the action with highest Q-value.
                If False, uses the decision function for action selection.
        
        Returns:
            A tuple containing:
                - action: The selected action as a tensor. (B, A)
                - info: A dictionary with 'q_value' key containing the Q-value of the selected action.
        """
        if obs.dim() == 2 and obs.shape[0] == 1:
            obs = obs.squeeze(0)  # Ensure obs is of shape (state,) for indexing

        action_values = self.get_action_values(obs)

        if not deterministic:
            action = self.decision_function.select_action(action_values)
        else:
            action = torch.argmax(action_values, dim=-1, keepdim=True)

        return action, {}#{'q_value': action_values[action]}
    
    def get_qvalue(self, obs: Tensor, action: Tensor) -> Tensor:
        """
        Get the Q-value for a specific state-action pair.
        
        Args:
            obs: The state/observation as a tensor.
            action: The action as a tensor.
        
        Returns:
            The Q-value for the given state-action pair. (1, )
        """
        return self.table[obs, action]

    def get_action_values(self, obs: Tensor) -> Tensor:
        """
        Get all action values (Q-values) for a given state.
        
        Args:
            obs: The state/observation as a tensor.
        
        Returns:
            A tensor containing Q-values for all possible actions in the given state. (action_dim, )
        """
        return self.table[obs]
    
    def set_qvalue(self, state: Tensor, action: Tensor, qval: Tensor):
        """
        Update the Q-value for a specific state-action pair.
        
        Args:
            state: The state as a tensor. (1,)
            action: The action as a tensor. (1,)
            qval: The new Q-value to set for the state-action pair. (1,)
        """
        self.table[state, action] = qval

class QLearningAgent(Agent):
    r"""
    Q-Learning trainer.

    .. math::
        Q(s,a)

    Args:
        env_params (EnvParams): environment parameters.
    """
    def __init__(self,
                 policy: QLearningPolicy,
                 config: QLearningConfig,
                 *,
                 device: str = "cpu",
                 ) -> None:
        super().__init__(device=device)
        self.policy = policy
        self.config = config

    @torch.no_grad()
    def act(self, obs, deterministic = False):
        """
        Select an action for the given observation.
        
        Args:
            obs: The current state/observation.
            deterministic: If True, selects the action with highest Q-value.
                If False, uses the policy's decision function for action selection.
        
        Returns:
            A tuple containing the selected action and additional information.
        """
        return self.policy.act(obs, deterministic)
    
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: Optional[List[ParameterScheduler]] = None,
              logger: Optional[Logger] = None,
              evaluator: Optional[Evaluator] = None,
              show_progress: bool = True
              ) -> None:
        """
        Train the Q-Learning agent in the given environment.
        
        Args:
            env: The environment to train in.
            total_steps: Total number of training steps to perform.
            schedulers: Optional list of parameter schedulers to update during training
                (e.g., for epsilon decay in epsilon-greedy policies).
            logger: Optional logger for recording training metrics. If None, creates a default logger.
            evaluator: Optional evaluator for periodic policy evaluation during training.
            show_progress: If True, displays a progress bar with training metrics.
        
        Returns:
            None. Updates the policy's Q-table in place.
        """
        
        logger = logger or Logger()

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)

        num_steps = 0

        # Make a collector and there is no need for a buffer because we will use the experience right away
        collector = Collector(env=env, logger=logger, flatten=True)

        while num_steps < total_steps:
            # Update Schedulers if provided
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.update(current_step=num_steps)

            experience = collector.collect_experience(policy=self.policy, num_steps=1)
            # Convert experience from (1, S, A) to (S, A) by squeezing the first dimension
            experience = {k: v.squeeze(0) for k, v in experience.items()}

            state = experience['state']
            action = experience['action']
            reward = experience['reward']
            next_state = experience['next_state']

            # Compute updated Q-Value
            qsa = self.policy.get_qvalue(state, action)
            qmax, _ = torch.max(self.policy.get_action_values(next_state), dim=-1, keepdim=True)
            qnew = qsa + self.config.alpha * (reward + self.config.gamma * qmax - qsa)

            deltaq = qnew - qsa

            # Update Policy
            self.policy.set_qvalue(state, action, qnew)
            num_steps += 1

            if show_progress:
                tracker = collector.get_metric_tracker()
                progress_bar.update(current_step=num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}, "
                                                                   f"Episode Length: {tracker.last_episode_length}, "
                                                                   f"Delta Q: {deltaq.cpu().item():.4f},")
                
            if logger.should_log(num_steps):
                logger.log_scalar("train/delta_q", deltaq, num_steps)

            if evaluator is not None:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)

        if evaluator is not None:
            evaluator.close()

    def _save_impl(self, path):
        """
        Writes a self-contained checkpoint directory.

        Layout:
          path/
            agent.pt
            policy.pt
        """
        path.mkdir(parents=True, exist_ok=True)

        payload = {
            "algo": "QLearning",
            "agent_format_version": 1,
            "config": asdict(self.config),
        }
        torch.save(payload, path / "agent.pt")
        self.policy.save(path / "policy.pt")

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "QLearningAgent":
        p = Path(path)
        agent_meta = torch.load(p / "agent.pt", map_location=map_location)

        if agent_meta["algo"] != "QLearning":
            raise ValueError(f"Loaded agent type {agent_meta['algo']} is not QLearning.")
        
        config = QLearningConfig(**agent_meta["config"])
        policy = QLearningPolicy.load(p / "policy.pt", map_location=map_location)

        return cls(policy=policy, config=config)        
