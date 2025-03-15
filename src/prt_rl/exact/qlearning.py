from tensordict.tensordict import TensorDict
import torch
from typing import Optional, List, Any
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.common.trainers import TDTrainer
from prt_rl.common.decision_functions import DecisionFunction
from prt_rl.common.policy import QTablePolicy
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.metrics import MetricTracker

class QLearning(TDTrainer):
    r"""
    Q-Learning trainer.

    .. math::
        Q(s,a)

    Args:
        env_params (EnvParams): environment parameters.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 num_envs: int = 1,
                 decision_function: Optional[DecisionFunction] = None,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 deterministic: bool = False,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 schedulers: Optional[List[ParameterScheduler]] = None,
                 ) -> None:
        self.deterministic = deterministic
        self.gamma = gamma
        self.alpha = alpha
        self.env_params = env.get_parameters()

        policy = QTablePolicy(
            env_params=self.env_params,
            num_envs=num_envs,
            decision_function=decision_function
        )
        super().__init__(env, policy, logger=logger, schedulers=schedulers, metric_tracker=metric_tracker)
        self.q_table = policy.get_qtable()

        # Log parameters if a logger is provided
        if logger is not None:
            self.logger.log_parameters({
                'gamma': self.gamma,
                'alpha': self.alpha,
            })

    def set_parameter(self,
                      name: str,
                      value: Any
                      ) -> None:
        try:
            if name == 'gamma':
                self.gamma = value
            elif name == 'alpha':
                self.alpha = value
            else:
                super().set_parameter(name, value)
        except ValueError:
            raise ValueError(f"Parameter '{name}' not found in QLearning.")

    def update_policy(self, experience: TensorDict) -> None:
        st = experience["observation"]
        at = experience["action"]
        st1 = experience['next', 'observation']
        rt1 = experience['next', 'reward']

        # Get the action with the maximum value for the next state
        q_max, _ = torch.max(self.q_table.get_action_values(st1), dim=1, keepdim=True)

        # Update Q table
        if self.deterministic:
            qval_update = rt1 + self.gamma*q_max
        else:
            qval = self.q_table.get_state_action_value(st, at)
            qval_update = qval + self.alpha * (rt1 + self.gamma*q_max - qval)

        self.q_table.update_q_value(state=st, action=at, q_value=qval_update)
