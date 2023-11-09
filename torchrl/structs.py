from dataclasses import dataclass, field


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    weight_decay: float
    scheduler: bool


@dataclass(frozen=True)
class LossConfig:
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    loss_critic_type: str
    entropy_eps: float
    critic_coef: float
    normalize_advantage: bool
    max_grad_norm: float
    mini_batch_size: int
    epochs: int


@dataclass(frozen=True)
class CollectorConfig:
    frame_skip: int
    frames_per_batch: int
    total_frames: int
    device: str = field(default="cpu")
    storing_device: str = field(default="cpu")
    max_frames_per_trajectory: float = field(default=-1)


@dataclass(frozen=True)
class PolicyConfig:
    num_cells: int
    depth: int


@dataclass(frozen=True)
class ValueConfig:
    num_cells: int
    depth: int


@dataclass(frozen=True)
class PPOConfig:
    device: str
    collector: CollectorConfig
    policy: PolicyConfig
    value: ValueConfig
    loss: LossConfig
    optim: OptimConfig


@dataclass(frozen=True)
class SimConfig:
    output_file: str
    environment: str
    ppo: PPOConfig
