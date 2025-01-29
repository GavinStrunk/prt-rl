import torch
from prt_rl.env.wrappers import GymnasiumWrapper
from prt_rl.ppo import PPO
from prt_rl.utils.runners import Runner
from prt_rl.utils.visualizers import PygameVisualizer

gym_name = 'CartPole-v1'
# gym_name = 'Pendulum-v1'
# gym_name = "MountainCar-v0"

env = GymnasiumWrapper(
    gym_name=gym_name,
)

torch.manual_seed(0)
trainer = PPO(
    env=env,
    # epsilon=0.2,
    # learning_rate=1e-3,
    # num_optim_steps=40,
    # mini_batch_size=128
)
trainer.train(num_episodes=10000)

eval_env = GymnasiumWrapper(
    gym_name=gym_name,
    render_mode="rgb_array",
)
runner = Runner(
    env=eval_env,
    policy=trainer.policy,
    visualizer=PygameVisualizer(
        fps=50,
    ),
)
runner.run()
