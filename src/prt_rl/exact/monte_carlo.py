from tensordict import TensorDict
from typing import Optional
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.decision_functions import DecisionFunction
from prt_rl.common.loggers import Logger
from prt_rl.common.policy import QTablePolicy
from prt_rl.common.trainers import TDTrainer
from prt_rl.common.metrics import MetricTracker

class MonteCarlo(TDTrainer):
    r"""
        On-policy First Visit Monte Carlo Algorithm

    .. math::
        \begin{equation}
        Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \frac{1}{N}[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1} - Q(S_t,A_t)] \\
        q_s \leftarrow q_s + \frac{1}{n}[G_t - q_s]
        \end{equation}
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 decision_function: Optional[DecisionFunction] = None,
                 gamma: float = 0.99,
                 logger: Optional[Logger] = None,
                 metric_tracker: Optional[MetricTracker] = None,
                 ) -> None:
        self.gamma = gamma
        self.env_params = env.get_parameters()

        policy = QTablePolicy(
            env_params=self.env_params,
            num_envs=1,
            decision_function=decision_function,
        )
        super(MonteCarlo, self).__init__(env=env, policy=policy, logger=logger, metric_tracker=metric_tracker)
        self.q_table = policy.get_qtable()

        # Log parameters if logger is provided
        if logger is not None:
            self.logger.log_parameters({
                'gamma': self.gamma,
            })

        # Initialize experience trajectory
        self.trajectory = []

    def update_policy(self, experience: TensorDict) -> None:
        # Add experience to trajectory
        self.trajectory.append(experience)

        # Return if a full episode has not completed
        if not experience['next', 'done']:
            return

        # Initialize return to zero
        G = 0

        # Learn by working through trajectory backwards
        for t in reversed(range(len(self.trajectory) - 1)):
            state = self.trajectory[t]['observation']
            action = self.trajectory[t]['action']
            reward = self.trajectory[t]['next', 'reward']

            # Update return
            G = self.gamma * G + reward

            # If this is a first visit update the visit and q tables
            if self._is_first_visit(t):
                self.q_table.update_visits(state=state, action=action)

                # Compute new Q value
                n = self.q_table.get_visit_count(state=state, action=action)
                qval = self.q_table.get_state_action_value(state=state, action=action)
                qval += 1/n * (G - qval)

                self.q_table.update_q_value(state=state, action=action, q_value=qval)

        # Reset the experience trajectory
        self.trajectory = []

    def _is_first_visit(self,
                        t: int
                        ) -> bool:
        """
        Checks if the state,action pair at timestep t in the trajectory is the first visit

        Args:
            t (int): Current timestep in the trajectory

        Returns:
            bool: True if this is the first visit, False otherwise
        """
        # If this is the first index then it is the first visit by default
        if t <= 0:
            return True

        # Get current state, action pair
        state = self.trajectory[t]['observation']
        action = self.trajectory[t]['action']

        # Check if this pair appears in any previous tensordicts
        for td in self.trajectory[:t]:
            if td.get('observation').equal(state) and td.get('action').equal(action):
                return False

        return True
