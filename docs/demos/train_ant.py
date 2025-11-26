from prt_rl.env.wrappers import IsaaclabWrapper
from prt_rl.common.loggers import FileLogger
from prt_rl.ppo import PPO, PPOConfig
from prt_rl.common.policies import ActorCriticPolicy


def train():
    device = 'cuda'
    env_name = "Isaac-Ant-Direct-v0"
    num_envs = 10

    logger = FileLogger(output_dir='log', logging_freq=10_000)

    env = IsaaclabWrapper(env_name=env_name, num_envs=num_envs)

    config = PPOConfig(
        steps_per_batch=1000,
        mini_batch_size=256,
    )

    policy = ActorCriticPolicy(env_params=env.get_parameters())
    agent = PPO(policy=policy, config=config, device=device)

    agent.train(env=env, total_steps=1_000_000, logger=logger)

    logger.close()

if __name__ == '__main__':
    train()