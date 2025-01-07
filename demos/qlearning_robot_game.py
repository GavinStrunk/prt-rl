from prt_sim.jhu.robot_game import RobotGame
from prt_rl.env.wrappers import JhuWrapper
from prt_rl.exact.qlearning import QLearning
from prt_rl.utils.loggers import MLFlowLogger
from prt_rl.utils.schedulers import LinearScheduler
from prt_rl.utils.policies import load_from_mlflow



# Parameters
tracking_uri = "http://127.0.0.1:5000"

env = JhuWrapper(environment=RobotGame())

schedulers = [
    LinearScheduler(parameter_name='epsilon', start_value=0.8, end_value=0.05, num_episodes=100),
    LinearScheduler(parameter_name='alpha', start_value=0.3, end_value=0.01, num_episodes=100),
]

trainer = QLearning(
    env=env,
    gamma=0.9,
    alpha=0.1,
    logger=MLFlowLogger(
        tracking_uri=tracking_uri,
        experiment_name="Robot Game",
        run_name="Q-Learning",
    ),
    schedulers=schedulers
)
trainer.train(num_episodes=100)
trainer.save_policy()

eval_env = JhuWrapper(environment=RobotGame(), render_mode="human")
# policy = trainer.get_policy()
policy = load_from_mlflow(tracking_uri=tracking_uri, model_name="Robot Game", model_version="2")
policy.set_parameter('epsilon', 1.0)

# Make evaluator to run the environment
done = False
state_td = eval_env.reset()
while not done:
    action_td = policy.get_action(state_td)
    state_td = eval_env.step(action_td)
    print(
        f"State: {state_td['observation']}  Action: {state_td['action']}  Next State: {state_td['next', 'observation']} Reward: {state_td['next', 'reward']}  Done: {state_td['next', 'done']}")

    done = state_td['next', 'done']

    # Update the MDP
    state_td = env.step_mdp(state_td)