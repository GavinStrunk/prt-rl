from collections import defaultdict
import hydra
from torch.nn import Sequential
from torchrl.modules import MLP

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from structs import SimConfig

class PPOAgent:
    def __init__(self, env, config: SimConfig):
        self.config = config
        self.env = env

        self.frames_per_batch = self.config.ppo.collector.frames_per_batch // self.config.ppo.collector.frame_skip
        self.total_frames = self.config.ppo.collector.total_frames // self.config.ppo.collector.frame_skip

        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                module=Sequential(
                    MLP(
                        in_features=env.observation_space.shape[0],
                        out_features=2 * env.action_spec.shape[0],
                        depth=3,
                        num_cells=self.config.ppo.policy.num_cells,
                        activate_last_layer=False,
                    ),
                    NormalParamExtractor(
                        # scale_mapping="biased_softplus_1.0",
                        # scale_lb=0.1,
                    ),
                ),
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            spec=env.action_spec,
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": env.action_spec.space.minimum,
                "max": env.action_spec.space.maximum,
                # "tanh_loc": False,
            },
            # default_interaction_type=ExplorationType.RANDOM,
            return_log_prob=True,
        ).to(self.config.ppo.device)

        self.value_module = ValueOperator(
            in_keys=["observation"],
            module=MLP(
                in_features=env.observation_space.shape[0],
                out_features=1,
                depth=3,
                num_cells=self.config.ppo.value.num_cells,
                activate_last_layer=False,
            ),
        ).to(self.config.ppo.device)

        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=self.config.ppo.device,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )

        self.advantage_module = GAE(
            gamma=self.config.ppo.loss.gamma, lmbda=self.config.ppo.loss.gae_lambda, value_network=self.value_module,
            average_gae=True
        )

        self.loss_module = ClipPPOLoss(
            actor=self.policy_module,
            critic=self.value_module,
            clip_epsilon=self.config.ppo.loss.clip_epsilon,
            entropy_bonus=bool(self.config.ppo.loss.entropy_eps),
            entropy_coef=self.config.ppo.loss.entropy_eps,
            # these keys match by default but we set this for completeness
            # value_target_key=self.advantage_module.value_target_key,
            critic_coef=self.config.ppo.loss.critic_coef,
            # gamma=self.config.ppo.loss.gamma,
            loss_critic_type=self.config.ppo.loss.loss_critic_type,
        )
        self.loss_module.make_value_estimator(gamma=self.config.ppo.loss.gamma)

        self.optim = torch.optim.Adam(self.loss_module.parameters(), self.config.ppo.optim.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, self.total_frames // self.frames_per_batch, 0.0
        )

    def learn(self):
        logs = defaultdict(list)
        pbar = tqdm(total=self.config.ppo.collector.total_frames * self.config.ppo.collector.frame_skip)
        eval_str = ""

        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(self.collector):
            # we now have a batch of data to work with. Let's learn something from it.
            for _ in range(self.config.ppo.loss.epochs):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                with torch.no_grad():
                    self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())
                for _ in range(self.config.ppo.collector.frames_per_batch // self.config.ppo.loss.mini_batch_size):
                    subdata = self.replay_buffer.sample(self.config.ppo.loss.mini_batch_size)
                    loss_vals = self.loss_module(subdata.to(self.config.ppo.device))
                    loss_value = (
                            loss_vals["loss_objective"]
                            + loss_vals["loss_critic"]
                            + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optim step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.config.ppo.loss.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel() * self.config.ppo.collector.frame_skip)
            cum_reward_str = (
                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            # logs["step_count"].append(tensordict_data["step_count"].max().item())
            # stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            logs["lr"].append(self.optim.param_groups[0]["lr"])
            lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (1000, which is our env horizon).
                # The ``rollout`` method of the env can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = self.env.rollout(1000, self.policy_module)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    # logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    )
                    del eval_rollout
            pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str]))

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            self.scheduler.step()

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()

    def save(self, file_name: str):
        torch.save(self.policy_module, file_name)


@hydra.main(config_path='configs', config_name='invert_double_pend_ppo.yaml', version_base=None)
def train(config: SimConfig):
    frame_skip = 1
    env = GymEnv(config.environment, device='cuda', frame_skip=frame_skip)

    env = TransformedEnv(
        env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(in_keys=["observation"]),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    # test_tutorial(env)
    model = PPOAgent(env=env, config=config)
    model.learn()
    model.save(file_name="models/invert_double_pend_ppo.zip")


@hydra.main(config_path='configs', config_name='invert_double_pend_ppo.yaml', version_base=None)
def execute(config: SimConfig):
    env = GymEnv(config.environment, device='cuda', render_mode="human")

    env = TransformedEnv(
        env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(in_keys=["observation"]),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    policy = torch.load("models/invert_double_pend_ppo.zip").to(config.ppo.device)
    tensor_out = env.reset()

    done = False
    while not done:
        action = policy(tensor_out)
        tensor_out = env.step(action)
        done = tensor_out['next']['done'].item()


if __name__ == '__main__':
    train()
    # execute()
