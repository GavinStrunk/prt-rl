from prt_sim.jhu.robot_game import RobotGame
from prt_rl.env.wrappers import JhuWrapper
from prt_rl.exact.qlearning import QLearning

env = JhuWrapper(environment=RobotGame())

trainer = QLearning(
    env=env,
    gamma=0.9,
    alpha=0.1
)
trainer.train(num_episodes=1000)

eval_env = JhuWrapper(environment=RobotGame(), render_mode="human")
policy = trainer.get_policy()
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