from tensordict import TensorDict
from typing import Optional
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.decision_functions import DecisionFunction
from prt_rl.common.loggers import Logger
from prt_rl.common.policy import QTablePolicy
from prt_rl.common.qtable import QTable
from prt_rl.common.trainers import TDTrainer
from prt_rl.common.metrics import MetricTracker

class SampleAverage(TDTrainer):
    r"""
    Sample average trainer.

    Sample averaging is the same as every visit Monte Carlo with a gamma value of 0.
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 decision_function: Optional[DecisionFunction] = None,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 ) -> None:
        self.env_params = env.get_parameters()

        qtable = QTable(
                state_dim=self.env_params.observation_max +1,
                action_dim=self.env_params.action_max +1,
                batch_size=1,
                initial_value=0.0,
                track_visits=True,
                device='cpu',
            )
        policy = QTablePolicy(
            env_params=self.env_params,
            num_envs=1,
            decision_function=decision_function,
            qtable=qtable,
        )
        super(SampleAverage, self).__init__(env=env, policy=policy, logger=logger, metric_tracker=metric_tracker)
        self.q_table = policy.get_qtable()

    def update_policy(self, experience: TensorDict) -> None:
        state = experience['observation']
        action = experience['action']
        reward = experience['next', 'reward']

        # Update the visit count
        self.q_table.update_visits(state=state, action=action)

        # Update the sample average
        n = self.q_table.get_visit_count(state=state, action=action)
        qval = self.q_table.get_state_action_value(state=state, action=action)
        qval += 1/n * (reward - qval)
        self.q_table.update_q_value(state=state, action=action, q_value=qval)

