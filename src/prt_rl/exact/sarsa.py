from tensordict.tensordict import TensorDict
from typing import Optional, List
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.trainers import TDTrainer
from prt_rl.common.decision_functions import DecisionFunction
from prt_rl.common.policy import QTablePolicy
from prt_rl.common.loggers import Logger
from prt_rl.common.metrics import MetricTracker
from prt_rl.common.schedulers import ParameterScheduler

class SARSA(TDTrainer):
    """
    SARSA algorithm.

    """
    def __init__(self,
                 env: EnvironmentInterface,
                 num_envs: int = 1,
                 decision_function: Optional[DecisionFunction] = None,
                 gamma: float = 0.99,
                 alpha: float = 0.1,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 schedulers: Optional[List[ParameterScheduler]] = None,
                 ) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.env_params = env.get_parameters()

        policy = QTablePolicy(
            env_params=self.env_params,
            num_envs=num_envs,
            decision_function=decision_function
        )
        super().__init__(env=env, policy=policy, logger=logger, schedulers=schedulers, metric_tracker=metric_tracker)
        self.q_table = policy.get_qtable()

    def update_policy(self, experience: TensorDict) -> None:
        st = experience["observation"]
        at = experience["action"]
        st1 = experience['next', 'observation']
        rt1 = experience['next', 'reward']

        # Select action for next state
        mdp1 = experience.clone()
        mdp1['observation'] = st1
        at1 = self.policy.get_action(mdp1)
        at1 = at1['action']

        # Compute update
        qval = self.q_table.get_state_action_value(st, at)
        qval1 = self.q_table.get_state_action_value(st1, at1)
        qval_update = qval + self.alpha * (rt1 + self.gamma*(qval1 - qval))

        # Update q table
        self.q_table.update_q_value(state=st, action=at, q_value=qval_update)
