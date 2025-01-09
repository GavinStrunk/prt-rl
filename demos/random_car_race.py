from prt_rl.utils.policies import RandomPolicy
from prt_rl.env.wrappers import GymnasiumWrapper

env = GymnasiumWrapper(
    gym_name="CarRacing-v3",
    render_mode="human",
    continuous=True,
)
policy = RandomPolicy(env_params=env.get_parameters())

state_td = env.reset()
done = False
while not done:
    action_td = policy.get_action(state_td)
    # print(action_td)
    state_td = env.step(action_td)
    done = state_td['next', 'done']
    # print(f"State: {state_td['observation']}  Action: {state_td['action']}  Next State: {state_td['next', 'observation']} Reward: {state_td['next', 'reward']}  Done: {state_td['next', 'done']}")

    # Update the MDP
    state_td = env.step_mdp(state_td)