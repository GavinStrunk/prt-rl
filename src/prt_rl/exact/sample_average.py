"""
Sample average algorithm
"""
import torch
from typing import List
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.loggers import Logger
from prt_rl.agent import BaseAgent
from prt_rl.common.collectors import SequentialCollector
from prt_rl.common.progress_bar import ProgressBar
from prt_rl.common.schedulers import ParameterScheduler
from prt_rl.common.evaluators import Evaluator
from prt_rl.common.policies import QTablePolicy


class SampleAverage(BaseAgent):
    """
    Sample average trainer.

    Sample averaging is the same as every visit Monte Carlo with a gamma value of 0.
    """
    def __init__(self,
                 policy: QTablePolicy,
                 device: str = 'cpu'
                 ) -> None:
        super().__init__()
        self.device = device
        self.policy = policy
        

    def predict(self, state, deterministic = False):
        """
        Perform an action based on the current state using the policy.

        Args:
            state (torch.Tensor): Current state of the environment.
            deterministic (bool): If True, use the deterministic action from the policy. Default is False.

        Returns:
            torch.Tensor: Action to be taken by the agent.
        """        
        with torch.no_grad():
            return self.policy.predict(state, deterministic)
    
    def train(self,
              env: EnvironmentInterface,
              total_steps: int,
              schedulers: List[ParameterScheduler] = [],
              logger: Logger | None = None,
              evaluator: Evaluator | None = None,
              show_progress: bool = True
              ) -> None:
        logger = logger or Logger.create('blank')

        if show_progress:
            progress_bar = ProgressBar(total_steps=total_steps)  

        # Set up collector for a single step
        collector = SequentialCollector(env=env, logger=logger)
        
        num_steps = 0
        while num_steps < total_steps:
            # Update schedulers if any
            for scheduler in schedulers:
                scheduler.update(current_step=num_steps)

            # Collect a single step of experience
            experience = collector.collect_experience(policy=self.policy, num_steps=1)
            state = experience['state']
            action = experience['action']
            reward = experience['reward']
            num_steps += state.shape[0]

            # Update visit count
            self.policy.update_visits(state=state, action=action)

            # Update sample average
            N = self.policy.get_visit_count(state=state, action=action)
            qval = self.policy.get_state_action_value(state=state, action=action)
            new_qval = qval + 1/N * (reward - qval)
            self.policy.update_q_value(state=state, action=action, q_value=new_qval)

            if show_progress:
                tracker = collector.get_metric_tracker()
                progress_bar.update(num_steps, desc=f"Episode Reward: {tracker.last_episode_reward:.2f}, "
                                                                   f"Episode Length: {tracker.last_episode_length}, ")

            if logger.should_log(num_steps):
                for scheduler in schedulers:
                    logger.log_scalar(name=scheduler.parameter_name, value=getattr(scheduler.obj, scheduler.parameter_name), iteration=num_steps)

            if evaluator is not None:
                evaluator.evaluate(agent=self.policy, iteration=num_steps)

        if evaluator is not None:
            evaluator.close()

