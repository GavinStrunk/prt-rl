from prt_rl.utils.policies import KeyboardPolicy
from prt_rl.env.wrappers import GymnasiumWrapper

env = GymnasiumWrapper(
    gym_name="CliffWalking-v0",
    render_mode="human"
)

policy = KeyboardPolicy(
    env_params=env.get_parameters(),
    key_action_map={
        "up": 0,
        'right': 1,
        'down': 2,
        'left': 3,
    },
    blocking=True
)

state_td = env.reset()
done = False
while not done:
    action_td = policy.get_action(state_td)
    state_td = env.step(action_td)
    done = state_td['next', 'done']
    print(
        f"State: {state_td['observation']}  Action: {state_td['action']}  Next State: {state_td['next', 'observation']} Reward: {state_td['next', 'reward']}  Done: {state_td['next', 'done']}")

    # Update the MDP
    state_td = env.step_mdp(state_td)
