import os
from prt_sim.jhu.gold_explorer import GoldExplorer
from prt_rl.env.wrappers import JhuWrapper
from prt_rl.exact.qlearning import QLearning
from prt_rl.exact.sarsa import SARSA
from prt_rl.dqn import DQN
from prt_rl.utils.loggers import MLFlowLogger
from prt_rl.utils.schedulers import LinearScheduler
from prt_rl.utils.runners import Runner
from prt_rl.utils.visualizers import PygameVisualizer

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def make_environment(training: bool = True):
    if training:
        env = JhuWrapper(environment=GoldExplorer())
    else:
        env = JhuWrapper(environment=GoldExplorer(), render_mode="rgb_array")
    return env

def train_qlearning(env):
    num_episodes = 1000
    schedulers = [
        LinearScheduler(parameter_name='epsilon', start_value=0.2, end_value=0.01, num_episodes=num_episodes),
    ]

    trainer = QLearning(
        env=env,
        gamma=0.9,
        alpha=0.1,
        logger=MLFlowLogger(
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
            experiment_name="Gold Explorer",
            run_name="Q-Learning",
        ),
        schedulers=schedulers
    )
    trainer.train(num_episodes=num_episodes, num_agents=10)
    trainer.save_policy()
    return trainer.get_policy()

def train_sarsa(env):
    num_episodes = 1000
    schedulers = [
        LinearScheduler(parameter_name='epsilon', start_value=0.2, end_value=0.01, num_episodes=num_episodes),
    ]

    trainer = SARSA(
        env=env,
        gamma=0.9,
        alpha=0.1,
        logger=MLFlowLogger(
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
            experiment_name="Gold Explorer",
            run_name="SARSA",
        ),
        schedulers=schedulers
    )
    trainer.train(num_episodes=num_episodes)
    trainer.save_policy()
    return trainer.get_policy()

def train_dqn(env):
    num_episodes = 10000
    schedulers = [
        LinearScheduler(parameter_name='epsilon', start_value=0.8, end_value=0.1, num_episodes=num_episodes),
    ]

    trainer = DQN(
        env=env,
        gamma=0.99,
        alpha=5e-4,
        logger=MLFlowLogger(
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
            experiment_name="Gold Explorer",
            run_name="DQN",
        ),
        schedulers=schedulers
    )
    trainer.train(num_episodes=num_episodes)
    trainer.save_policy()
    return trainer.get_policy()

def main():
    env = make_environment()

    # policy = train_qlearning(env)
    # policy = train_sarsa(env)
    policy = train_dqn(env)

    eval_env = make_environment(training=False)
    runner = Runner(
        env=eval_env,
        policy=policy,
        visualizer=PygameVisualizer(fps=5)
    )
    runner.run()

if __name__ == "__main__":
    main()