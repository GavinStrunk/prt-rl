import flappy_bird_gymnasium
from prt_rl.env.wrappers import GymnasiumWrapper
from prt_rl.utils.policies import KeyboardPolicy

def run_environment(env, policy, num_runs:int = 1):
    for _ in range(num_runs):
        done = False
        obs_td = env.reset()
        while not done:
            action_td = policy.get_action(obs_td)
            obs_td = env.step(action_td)
            done = obs_td['next','done']
            obs_td = env.step_mdp(obs_td)


def human_play():
    env = GymnasiumWrapper(
        gym_name="FlappyBird-v0",
        render_mode="human",
        use_lidar=False,
        normalize_obs=True
    )

    policy = KeyboardPolicy(
        env_params=env.get_parameters(),
        key_action_map={
            "up": 1
        },
        blocking=False
    )

    run_environment(env, policy)

def train_dqn():
    env = GymnasiumWrapper(
        gym_name="FlappyBird-v0",
        use_lidar=False,
        normalize_obs=True
    )


if __name__ == '__main__':
    human_play()

