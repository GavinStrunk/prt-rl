from pathlib import Path

import torch
from torch import nn

from prt_rl.common.buffers import ReplayBuffer
from prt_rl.common.components.heads import GaussianHead
from prt_rl.imitation.dagger import DAggerAgent, DAggerConfig, DAggerPolicy


class TestDAggerPolicy(DAggerPolicy):
    """A concrete DAggerPolicy subclass with no required arguments for load compatibility."""
    def __init__(self):
        network = nn.Linear(4, 8)
        head = GaussianHead(latent_dim=8, action_dim=2)
        super().__init__(network=network, distribution_head=head)


def _build_policy() -> DAggerPolicy:
    return TestDAggerPolicy()


def _build_expert(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(4, 2)


def _build_replay_buffer(seed: int) -> ReplayBuffer:
    torch.manual_seed(seed)
    buffer = ReplayBuffer(capacity=16)
    buffer.add({
        "state": torch.randn(8, 4),
        "action": torch.randn(8, 2),
    })
    return buffer


def test_dagger_load_with_matching_objects(tmp_path: Path) -> None:
    expert = _build_expert(seed=0)
    replay_buffer = _build_replay_buffer(seed=1)
    agent = DAggerAgent(
        expert_policy=expert,
        experience_buffer=replay_buffer,
        policy=_build_policy(),
        config=DAggerConfig(),
    )

    ckpt_dir = tmp_path / "dagger_ckpt"
    agent.save(ckpt_dir)

    loaded = DAggerAgent.load(
        ckpt_dir,
        expert_policy=expert,
        experience_buffer=replay_buffer,
        policy_cls=TestDAggerPolicy,
    )
    assert isinstance(loaded, DAggerAgent)

def test_dagger_load_with_new_expert_and_buffer(tmp_path: Path) -> None:
    expert = _build_expert(seed=0)
    replay_buffer = _build_replay_buffer(seed=1)
    agent = DAggerAgent(
        expert_policy=expert,
        experience_buffer=replay_buffer,
        policy=_build_policy(),
        config=DAggerConfig(),
    )

    ckpt_dir = tmp_path / "dagger_ckpt"
    agent.save(ckpt_dir)

    # Load with fresh expert and buffer instances
    new_expert = _build_expert(seed=0)
    new_replay_buffer = _build_replay_buffer(seed=1)

    loaded = DAggerAgent.load(
        ckpt_dir,
        expert_policy=new_expert,
        experience_buffer=new_replay_buffer,
        policy_cls=TestDAggerPolicy,
    )
    assert isinstance(loaded, DAggerAgent)
    assert isinstance(loaded, DAggerAgent)
