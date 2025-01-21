from prt_rl.env.wrappers import GymnasiumWrapper
from prt_rl.utils.policy import ActorCriticPolicy
from prt_rl.ppo import PPO
from prt_rl.utils.runners import Runner
from prt_rl.utils.visualizers import PygameVisualizer

gym_name = 'CartPole-v1'
# gym_name = 'Pendulum-v1'

env = GymnasiumWrapper(
    gym_name=gym_name,
)

trainer = PPO(
    env=env,
)
trainer.train(num_episodes=200)

eval_env = GymnasiumWrapper(
    gym_name=gym_name,
    render_mode="rgb_array",
)
runner = Runner(
    env=eval_env,
    policy=trainer.policy,
    visualizer=PygameVisualizer(
        fps=10,
    ),
)
runner.run()
