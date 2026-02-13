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
class SARSAConfig:
    """
    Configuration parameters for the SARSA algorithm.
    
    Attributes:
        gamma: Discount factor for future rewards. A value between 0 and 1 that determines
            how much the agent values future rewards compared to immediate rewards.
            Default is 0.99.
        alpha: Learning rate for Q-value updates. Controls how much the Q-values are
            adjusted based on new experience. Default is 0.1.
    """
    gamma: float = 0.99
    alpha: float = 0.1

class SARSAPolicy(TabularPolicy):
    """
    SARSA Policy implementation using a tabular Q-table.
    
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
            The Q-value for the given state-action pair.
        """
        return self.table[obs, action]

    def get_action_values(self, obs: Tensor) -> Tensor:
        """
        Get all action values (Q-values) for a given state.
        
        Args:
            obs: The state/observation as a tensor.
        
        Returns:
            A tensor containing Q-values for all possible actions in the given state.
        """
        return self.table[obs]
    
    def set_qvalue(self, state: Tensor, action: Tensor, qval: Tensor):
        """
        Update the Q-value for a specific state-action pair.
        
        Args:
            state: The state as a tensor.
            action: The action as a tensor.
            qval: The new Q-value to set for the state-action pair.
        """
        self.table[state, action] = qval

class SARSAAgent(Agent):
    """
    SARSA (State-Action-Reward-State-Action) on-policy temporal difference control algorithm.
    
    SARSA is an on-policy reinforcement learning algorithm that learns action-values Q(s,a)
    by updating them based on the next action actually taken by the current policy. The update
    rule is: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)], where (s,a,r,s',a') is the SARSA tuple.
    
    Args:
        policy: The SARSA policy containing the Q-table and decision function.
        config: Configuration parameters including learning rate (alpha) and discount factor (gamma).
        device: The device to use for computations ("cpu" or "cuda"). Default is "cpu".
    
    Attributes:
        policy: The SARSA policy instance.
        config: The configuration parameters.
    """
    def __init__(self,
                 policy: SARSAPolicy,
                 config: SARSAConfig,
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
        Train the SARSA agent in the given environment.
        
        The agent performs on-policy learning by collecting experience one step at a time,
        computing the next action according to the current policy, and updating Q-values
        using the SARSA update rule: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)].
        
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

            # Get the next action according the current policy
            next_action, _ = self.policy.act(next_state)

            # Compute updated Q-Value
            qsa = self.policy.get_qvalue(state, action)
            qsa_next = self.policy.get_qvalue(next_state, next_action)
            qnew = qsa + self.config.alpha * (reward + self.config.gamma * (qsa_next - qsa))
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
            "algo": "SARSA",
            "agent_format_version": 1,
            "config": asdict(self.config),
        }
        torch.save(payload, path / "agent.pt")
        self.policy.save(path / "policy.pt")

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "SARSAAgent":
        p = Path(path)
        agent_meta = torch.load(p / "agent.pt", map_location=map_location)

        if agent_meta["algo"] != "SARSA":
            raise ValueError(f"Loaded agent type {agent_meta['algo']} is not SARSA.")
        
        config = SARSAConfig(**agent_meta["config"])
        policy = SARSAPolicy.load(p / "policy.pt", map_location=map_location)

        return cls(policy=policy, config=config)