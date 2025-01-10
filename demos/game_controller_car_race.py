from prt_rl.utils.policy import GameControllerPolicy
from prt_rl.env.wrappers import GymnasiumWrapper

env = GymnasiumWrapper(
    gym_name="CarRacing-v3",
    render_mode="human",
    continuous=True,
)
policy = GameControllerPolicy(
    env_params=env.get_parameters(),
    key_action_map={
        GameControllerPolicy.Key.JOYSTICK_RIGHT_X: 0,
        GameControllerPolicy.Key.JOYSTICK_LEFT_Y: (1, 'positive'),
        GameControllerPolicy.Key.JOYSTICK_RIGHT_Y: (2, 'negative'),
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
