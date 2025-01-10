from prt_rl.utils.policy import RandomPolicy
from prt_rl.env.wrappers import GymnasiumWrapper
from prt_rl.utils.runners import Runner
from prt_rl.utils.recorders import GifRecorder
from prt_rl.utils.visualizers import PygameVisualizer

env = GymnasiumWrapper(
    gym_name="CarRacing-v3",
    render_mode="rgb_array",
    continuous=True,
)
policy = RandomPolicy(env_params=env.get_parameters())

runner = Runner(
    env=env,
    policy=policy,
    # recorder=GifRecorder(
    #     filename="car_race.gif",
    #     fps=50
    # ),
    visualizer=PygameVisualizer(
        fps=50,
        caption="Car Racing",
    ),
)
runner.run()

# state_td = env.reset()
# done = False
# while not done:
#     action_td = policy.get_action(state_td)
#     # print(action_td)
#     state_td = env.step(action_td)
#     done = state_td['next', 'done']
#     # print(f"State: {state_td['observation']}  Action: {state_td['action']}  Next State: {state_td['next', 'observation']} Reward: {state_td['next', 'reward']}  Done: {state_td['next', 'done']}")
#
#     # Update the MDP
#     state_td = env.step_mdp(state_td)