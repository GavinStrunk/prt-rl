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
class MonteCarloConfig:
    gamma: float = 0.99

class MonteCarloPolicy(TabularPolicy):
    """
    Monte Carlo Policy implementation using a tabular Q-table.
    
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
    
    def set_qvalue(self, obs: Tensor, action: Tensor, qval: Tensor):
        """
        Update the Q-value for a specific state-action pair.
        
        Args:
            obs: The state/observation as a tensor. (1,)
            action: The action as a tensor. (1,)
            qval: The new Q-value to set for the state-action pair. (1,)
        """
        self.table[obs, action] = qval


class MonteCarloAgent(Agent):
    r"""
        On-policy First Visit Monte Carlo Algorithm

    .. math::
        \begin{equation}
        Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \frac{1}{N}[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1} - Q(S_t,A_t)] \\
        q_s \leftarrow q_s + \frac{1}{n}[G_t - q_s]
        \end{equation}
    """
    def __init__(self,
                 policy: MonteCarloPolicy,
                 config: MonteCarloConfig,
                 *,
                 device: str = "cpu",
                 ) -> None:
        super().__init__(device=device)
        self.policy = policy
        self.config = config
        self.visit_table = torch.zeros_like(self.policy.table, dtype=torch.long, device=self.policy.table.device)

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
        Train the Monte Carlo agent in the given environment.
        
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

            # Collect a full trajectory (episode) using the current policy with shape (B, ...)
            trajectory = collector.collect_trajectory(policy=self.policy, num_trajectories=1)        
            num_steps += trajectory['state'].shape[0]

            # Initialize return to zero
            G = 0
            mask = self._first_visit_mask(trajectory['state'], trajectory['action'], num_actions=self.policy.table.shape[1])

            for t in reversed(range(len(trajectory['state']) - 1)):
                state = trajectory['state'][t]
                action = trajectory['action'][t]
                reward = trajectory['reward'][t]

                # Update return
                G = self.config.gamma * G + reward

                if mask[t]:
                    # Increment the visits table
                    self._update_visit_count(state=state, action=action)

                    # Compute new Q value
                    n = self._get_visit_count(state=state, action=action)
                    qval = self.policy.get_qvalue(state, action)
                    qnew = qval + 1/n * (G - qval)

                    self.policy.set_qvalue(state, action, qnew)

            if show_progress:
                tracker = collector.get_metric_tracker()
                progress_bar.update(current_step=num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}, "
                                                                   f"Episode Length: {tracker.last_episode_length}, ")
                
            if logger.should_log(num_steps):
                pass

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
            "algo": "MonteCarlo",
            "agent_format_version": 1,
            "config": asdict(self.config),
            "visit_table": self.visit_table.cpu(),  
        }
        torch.save(payload, path / "agent.pt")
        self.policy.save(path / "policy.pt")

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "MonteCarloAgent":
        p = Path(path)
        agent_meta = torch.load(p / "agent.pt", map_location=map_location)

        if agent_meta["algo"] != "MonteCarlo":
            raise ValueError(f"Loaded agent type {agent_meta['algo']} is not MonteCarlo.")
        
        config = MonteCarloConfig(**agent_meta["config"])
        policy = MonteCarloPolicy.load(p / "policy.pt", map_location=map_location)

        visit_table = agent_meta["visit_table"].to(policy.table.device)

        # Update the agent's visit table with the loaded visit table
        agent = cls(policy=policy, config=config)
        agent.visit_table = visit_table
        return agent 

    def _get_visit_count(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Get the visit count for a specific state-action pair.
        
        Args:
            state: The state as a tensor. (1,)
            action: The action as a tensor. (1,)
        Returns:
            The number of times the state-action pair has been visited.
        """
        return self.visit_table[state, action]
    
    def _update_visit_count(self, state: Tensor, action: Tensor):
        """
        Increment the visit count for a specific state-action pair.
        
        Args:
            state: The state as a tensor. (1,)
            action: The action as a tensor. (1,)
        """
        self.visit_table[state, action] += 1
    
    @staticmethod
    def _first_visit_mask(states: Tensor, actions: Tensor, num_actions: int) -> Tensor:
        """
        Compute a boolean mask indicating the first visit of each state-action pair in the trajectory.

        Args:
            states: Tensor of shape (T, state_dim) containing the states in the trajectory.
            actions: Tensor of shape (T, action_dim) containing the actions in the trajectory.
            num_actions: The total number of possible actions.

        Returns:
            A boolean tensor of shape (T,) where True indicates the first visit of a state-action pair.
        """
        # key = s * n_actions + a
        key = states.to(torch.int64) * int(num_actions) + actions.to(torch.int64)  # [T]
        key = key.reshape(-1)  # Ensure key is 1D

        # unique gives inverse mapping; we want first index for each unique value
        uniq, inv = torch.unique(key, return_inverse=True)
        first_idx = torch.full((uniq.numel(),), fill_value=key.numel(), device=key.device, dtype=torch.long)
        first_idx.scatter_reduce_(0, inv, torch.arange(key.numel(), device=key.device), reduce="amin", include_self=True)

        mask = torch.zeros_like(key, dtype=torch.bool)
        mask[first_idx] = True
        return mask
