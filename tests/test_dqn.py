import torch
from prt_rl.common.policies import QValuePolicy
from prt_rl.env.interface import EnvParams
from prt_rl.dqn import DQN


def test_polyak_update():
    # Dummy environment parameters
    env_params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=3,
        observation_shape=(3,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )

    # Create policy and target networks
    policy = QValuePolicy(env_params)
    target = QValuePolicy(env_params)

    # Modify policy weights to ensure they're different from target
    for param in policy.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param))

    # Store copies of original parameters for comparison
    policy_params_before = [p.clone().detach() for p in policy.parameters()]
    target_params_before = [p.clone().detach() for p in target.parameters()]

    tau = 0.1  # Polyak coefficient
    DQN._polyak_update(policy, target, tau)

    # Check that updated target is a convex combination
    for i, (p_old, t_old, t_new) in enumerate(zip(policy_params_before, target_params_before, target.parameters())):
        expected = tau * p_old + (1 - tau) * t_old
        assert torch.allclose(t_new, expected, atol=1e-6), f"Polyak update failed on param {i}"

def test_hard_update():
    # Dummy environment parameters
    env_params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=3,
        observation_shape=(3,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )

    # Create two identical policies
    policy = QValuePolicy(env_params)
    target = QValuePolicy(env_params)

    # Make target's weights different from policy
    for param in target.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param))

    # Ensure they are initially different
    different = any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(policy.parameters(), target.parameters())
    )
    assert different, "Test invalid: policy and target should start with different weights"

    # Copy target into policy
    DQN._hard_update(policy, target)

    # Now they should match exactly
    for p1, p2 in zip(policy.parameters(), target.parameters()):
        assert torch.allclose(p1, p2), "Parameter mismatch after hard update"

    # Also check buffers if needed (e.g., BatchNorm)
    for b1, b2 in zip(policy.buffers(), target.buffers()):
        assert torch.allclose(b1, b2), "Buffer mismatch after hard update"

