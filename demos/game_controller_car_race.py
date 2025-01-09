from prt_rl.utils.policies import GameControllerPolicy
from prt_rl.env.wrappers import GymnasiumWrapper

env = GymnasiumWrapper(
    gym_name="CarRacing-v3",
    render_mode="human",
    continuous=True,
)
policy = GameControllerPolicy(
    env_params=env.get_parameters(),
    key_action_map={
        'JOY_RIGHT_X': 0,
        'JOY_LEFT_Y': 1,
        'JOY_R2': 2,
    },
    blocking=False
)

state_td = env.reset()
done = False
while not done:
    action_td = policy.get_action(state_td)
    print(action_td['action'][0])
    state_td = env.step(action_td)
    done = state_td['next', 'done']

    # Update the MDP
    state_td = env.step_mdp(state_td)