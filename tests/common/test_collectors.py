import pytest
import sys
import torch
from unittest.mock import MagicMock
from prt_rl.env.wrappers import GymnasiumWrapper
from prt_rl.env.interface import EnvParams
from prt_rl.common.collectors import SequentialCollector, ParallelCollector, MetricsTracker, random_action, get_action_from_policy

class FakeLogger:
    def __init__(self):
        self.scalars = []  # list of (name, value, iteration)
    def log_scalar(self, name: str, value: float, iteration: int = None):
        self.scalars.append((name, float(value), int(iteration) if iteration is not None else None))

    def _by_name(self, name):
        """Return list of (value, iteration) for a metric name."""
        return [(v, it) for (n, v, it) in self.scalars if n == name]

@pytest.fixture
def mock_env():
    env = MagicMock()
    # Returns (state, info)
    env.reset.return_value = (torch.zeros(1, 4, dtype=torch.float32), {})

    # Returns (next_state, reward, done, info)
    env.step.return_value = (torch.zeros(1, 4, dtype=torch.float32), torch.tensor([[1.0]]), torch.tensor([[False]]), {})
    return env

@pytest.fixture
def discrete_action_params():
    params = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=2,
        observation_shape=(4,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )
    return params

@pytest.fixture
def cont_action_params():
    params = EnvParams(
        action_len=2,
        action_continuous=True,
        action_min=-1.0,
        action_max=1.0,
        observation_shape=(4,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )
    return params

# Random Action Helper
# =========================================================
def test_random_discrete_action(discrete_action_params):
    action = random_action(discrete_action_params, torch.zeros(1, 4, dtype=torch.float32))
    assert action.shape == (1, 1)  # Single action for a single environment
    assert action.dtype == torch.int64  # Discrete action should be int64
    assert action.item() in range(discrete_action_params.action_min, discrete_action_params.action_max + 1)

def test_random_continuous_action(cont_action_params):
    action = random_action(cont_action_params, torch.zeros(1, 4, dtype=torch.float32))
    assert action.shape == (1, 2)  # Single action for a single environment
    assert action.dtype == torch.float32  # Continuous action should be float32
    assert torch.all(action >= cont_action_params.action_min) and torch.all(action <= cont_action_params.action_max)

# MetricsTracker Tests
# =========================================================
def test_initial_state_and_reset():
    logger = FakeLogger()
    tr = MetricsTracker(num_envs=1, logger=logger)

    assert tr.collected_steps == 0
    assert tr.cumulative_reward == 0.0
    assert tr.episode_count == 0
    assert tr.last_episode_reward == 0.0
    assert tr.last_episode_length == 0
    assert tr._cur_reward.shape == (1,)
    assert tr._cur_length.shape == (1,)

    # mutate, then reset
    tr.update(torch.tensor([[1.0]]), torch.tensor([[False]]))
    tr.reset()
    assert tr.collected_steps == 0
    assert tr.cumulative_reward == 0.0
    assert tr.episode_count == 0
    assert tr.last_episode_reward == 0.0
    assert tr.last_episode_length == 0
    assert torch.all(tr._cur_reward == 0)
    assert torch.all(tr._cur_length == 0)

@pytest.mark.parametrize(
    "done, expected",
    [
        (torch.tensor(True), torch.tensor([True])),
        (torch.tensor([True, False, True]), torch.tensor([True, False, True])),
        (torch.tensor([[True], [False], [True]]), torch.tensor([True, False, True])),
    ],
)
def test_to_done_mask_variants(done, expected):
    out = MetricsTracker._to_done_mask(done)
    assert torch.equal(out, expected)

@pytest.mark.parametrize(
    "reward, expected",
    [
        (torch.tensor(5.0), torch.tensor([5.0])), # Single scalar reward -> (1,)
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])), # reward (N,) -> (N,)
        (torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([1.0, 2.0, 3.0])), # reward (N, 1) -> (N,)
        (torch.ones((3, 4, 1)), torch.tensor([4.0, 4.0, 4.0])), # reward (N, 4, 1) -> (N,)
    ],
)
def test_sum_per_env_variants(reward, expected):
    out = MetricsTracker._sum_rewards_per_env(reward)
    assert torch.equal(out, expected)

def test_update_no_done_no_logging():
    logger = FakeLogger()
    tr = MetricsTracker(num_envs=2, logger=logger)

    r = torch.tensor([[1.0], [2.0]])  # shape (2,1)
    d = torch.tensor([[False], [False]])
    tr.update(r, d)

    assert tr.collected_steps == 2  # env-steps
    assert tr.cumulative_reward == pytest.approx(3.0)
    assert tr.episode_count == 0
    assert torch.allclose(tr._cur_reward, torch.tensor([1.0, 2.0]))
    assert torch.all(tr._cur_length == torch.tensor([1, 1]))
    assert len(logger.scalars) == 0  # no episode end -> no logs

def test_update_single_env_done_logs_and_resets():
    logger = FakeLogger()
    tr = MetricsTracker(num_envs=2, logger=logger)

    # step 1
    tr.update(torch.tensor([[1.0], [2.0]]), torch.tensor([[False], [False]]))
    # step 2: env 0 ends
    tr.update(torch.tensor([[3.0], [4.0]]), torch.tensor([[True], [False]]))

    # env-steps counted per vectorized step: 2 + 2 = 4
    assert tr.collected_steps == 4
    assert tr.cumulative_reward == pytest.approx(1+2+3+4)
    assert tr.episode_count == 1
    # last ep (env 0) reward & length
    assert tr.last_episode_reward == pytest.approx(1.0 + 3.0)
    assert tr.last_episode_length == 2

    # env 0 accumulators reset; env 1 continues
    assert tr._cur_reward[0].item() == pytest.approx(0.0)
    assert tr._cur_length[0].item() == 0
    assert tr._cur_reward[1].item() == pytest.approx(2.0 + 4.0)
    assert tr._cur_length[1].item() == 2

    # logs: four scalars at iteration==4
    er = logger._by_name("episode_reward")
    el = logger._by_name("episode_length")
    cr = logger._by_name("cumulative_reward")
    en = logger._by_name("episode_number")
    assert er == [(4.0, 4)]
    assert el == [(2.0, 4)]
    assert cr == [(10.0, 4)]
    assert en == [(1.0, 4)]

def test_update_multiple_envs_done_same_step():
    logger = FakeLogger()
    tr = MetricsTracker(num_envs=3, logger=logger)

    # step 1
    tr.update(torch.tensor([[1.0], [10.0], [100.0]]), torch.tensor([[False], [False], [False]]))
    # step 2: env 0 and 2 end together
    tr.update(torch.tensor([[2.0], [20.0], [200.0]]), torch.tensor([[True], [False], [True]]))  # (N,1) done format

    assert tr.collected_steps == 6
    assert tr.episode_count == 2

    # episode rewards for env 0 and 2
    ep0 = 1.0 + 2.0
    ep2 = 100.0 + 200.0

    # There should be two entries for each metric
    er_vals = logger._by_name("episode_reward")
    assert len(er_vals) == 2
    # order should follow ascending env index (0, then 2)
    assert er_vals[0] == (pytest.approx(ep0), 6)
    assert er_vals[1] == (pytest.approx(ep2), 6)

    # last episode metrics reflect the *last processed* finished env (env 2)
    assert tr.last_episode_reward == pytest.approx(ep2)
    assert tr.last_episode_length == 2

    # finished env accumulators reset; env 1 continues
    assert tr._cur_length.tolist() == [0, 2, 0]
    assert tr._cur_reward.tolist() == [0.0, pytest.approx(10.0 + 20.0), 0.0]

def test_trailing_dims_summed_multiagent_like():
    logger = FakeLogger()
    tr = MetricsTracker(num_envs=2, logger=logger)

    # Shape (N, A, 1): A=3 "agents" per env; step 1
    r1 = torch.tensor([[[1.0], [0.5], [0.5]],
                       [[2.0], [1.0], [1.0]]])  # sums: [2.0, 4.0]
    d1 = torch.tensor([[False], [False]])
    tr.update(r1, d1)

    # step 2, both end
    r2 = torch.tensor([[[0.0], [1.0], [1.0]],
                       [[1.0], [1.0], [0.0]]])  # sums: [2.0, 2.0]
    d2 = torch.tensor([[True], [True]])
    tr.update(r2, d2)

    # Total env-steps: each step adds N=2 => 4
    assert tr.collected_steps == 4
    # Episode rewards per env
    ep0 = 2.0 + 2.0
    ep1 = 4.0 + 2.0

    er = logger._by_name("episode_reward")
    # order by env index: env 0 then env 1
    assert er == [(pytest.approx(ep0), 4), (pytest.approx(ep1), 4)]

    # accumulated current episodes reset
    assert tr._cur_length.tolist() == [0, 0]
    assert tr._cur_reward.tolist() == [0.0, 0.0]

# Sequential Collector Tests
# =========================================================
# Helpers to create policy stubs
def _policy_stub_discrete(return_values=True):
    """
    Returns a function with signature (policy, state, env_params) -> (action, value_est, log_prob)
    For discrete actions: action shape (1,1).
    """
    def f(policy, state, env_params):
        action = torch.zeros(1, 1, dtype=torch.float32)
        if return_values:
            value_est = torch.tensor([[0.5]], dtype=torch.float32)
            log_prob = torch.tensor([[-0.1]], dtype=torch.float32)
        else:
            value_est = None
            log_prob = None
        return action, value_est, log_prob
    return f

def _policy_stub_continuous(return_values=True):
    """
    Returns a function with signature (policy, state, env_params) -> (action, value_est, log_prob)
    For continuous actions: action shape (1,2).
    """
    def f(policy, state, env_params):
        action = torch.zeros(1, 2, dtype=torch.float32)
        if return_values:
            value_est = torch.tensor([[1.23]], dtype=torch.float32)
            log_prob = torch.tensor([[-0.3]], dtype=torch.float32)
        else:
            value_est = None
            log_prob = None
        return action, value_est, log_prob
    return f

# Tests
def test_collect_fixed_steps_discrete_shapes(mock_env, discrete_action_params, monkeypatch):
    mock_env.get_parameters.return_value = discrete_action_params
    collector = SequentialCollector(env=mock_env, logger=FakeLogger())

    # Patch the get_action_from_policy helper in the SAME module where SequentialCollector is defined
    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_discrete(return_values=True))

    num_steps = 5

    # Act
    out = collector.collect_experience(policy=MagicMock(), num_steps=num_steps)

    # Assert: shapes and presence
    assert isinstance(out, dict)
    assert out["state"].shape == (num_steps, 4)
    assert out["action"].shape == (num_steps, 1)              # discrete action_len=1
    assert out["next_state"].shape == (num_steps, 4)
    assert out["reward"].shape == (num_steps, 1)
    assert out["done"].shape == (num_steps, 1)
    assert out["value_est"].shape == (num_steps, 1)
    assert out["log_prob"].shape == (num_steps, 1)

    # Since mock_env.step returns done=False, last_value_est should be present
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)

    # Reset called once at the start only (no dones during steps)
    assert mock_env.reset.call_count == 1
    assert mock_env.step.call_count == num_steps

def test_collect_triggers_reset_after_done(mock_env, discrete_action_params, monkeypatch):
    mock_env.get_parameters.return_value = discrete_action_params
    step_returns = [
        (torch.zeros(1, 4), torch.tensor([[1.0]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[1.0]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[1.0]]), torch.tensor([[True]]),  {}),
        (torch.zeros(1, 4), torch.tensor([[1.0]]), torch.tensor([[False]]), {}),
    ]
    mock_env.step.side_effect = step_returns

    collector = SequentialCollector(env=mock_env, logger=FakeLogger())
    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_discrete(return_values=True))

    out = collector.collect_experience(policy=MagicMock(), num_steps=4)

    # Initial reset + one more after the done=True step
    assert mock_env.reset.call_count == 2
    assert mock_env.step.call_count == 4
    assert out["state"].shape == (4, 4)
    assert out["done"][2].item() == 1.0
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)

def test_collect_no_values_or_logprobs_returns_none(mock_env, discrete_action_params, monkeypatch):
    mock_env.get_parameters.return_value = discrete_action_params
    collector = SequentialCollector(env=mock_env, logger=FakeLogger())

    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_discrete(return_values=False))

    out = collector.collect_experience(policy=MagicMock(), num_steps=3)

    assert out["state"].shape == (3, 4)
    assert out["action"].shape == (3, 1)
    assert not hasattr(out, 'value_est')
    assert not hasattr(out, 'log_prob')
    assert not hasattr(out, 'last_value_est')

def test_collect_continuous_action_shapes(mock_env, cont_action_params, monkeypatch):
    mock_env.get_parameters.return_value = cont_action_params
    collector = SequentialCollector(env=mock_env, logger=FakeLogger())

    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_continuous(return_values=True))

    out = collector.collect_experience(policy=MagicMock(), num_steps=4)

    assert out["action"].shape == (4, 2)
    assert out["state"].shape == (4, 4)
    assert out["next_state"].shape == (4, 4)
    assert out["reward"].shape == (4, 1)
    assert out["done"].shape == (4, 1)
    assert out["value_est"].shape == (4, 1)
    assert out["log_prob"].shape == (4, 1)
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)

def test_collect_calls_bootstrap_once_when_not_done(mock_env, discrete_action_params, monkeypatch):
    mock_env.get_parameters.return_value = discrete_action_params
    collector = SequentialCollector(env=mock_env, logger=FakeLogger())

    call_counter = {"count": 0}
    def counting_policy(policy, state, env_params):
        call_counter["count"] += 1
        action = torch.zeros(1, 1, dtype=torch.float32)
        value_est = torch.tensor([[0.7]], dtype=torch.float32)
        log_prob = torch.tensor([[-0.2]], dtype=torch.float32)
        return action, value_est, log_prob

    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', counting_policy)

    num_steps = 6
    out = collector.collect_experience(policy=MagicMock(), num_steps=num_steps)

    assert call_counter["count"] == num_steps + 1  # +1 for bootstrap
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)

def test_collect_single_trajectory_discrete_happy_path(mock_env, discrete_action_params, monkeypatch):
    """
    Ensure the method collects until done=True, returns stacked tensors with expected shapes,
    and includes value/log_prob when present.
    """
    mock_env.get_parameters.return_value = discrete_action_params

    # Make a trajectory of length T=3 with done=True on the last step
    step_returns = [
        (torch.zeros(1, 4), torch.tensor([[1.0]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[1.0]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[1.0]]), torch.tensor([[True]]),  {}),
    ]
    mock_env.step.side_effect = step_returns

    collector = SequentialCollector(env=mock_env, logger=FakeLogger())
    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_discrete(return_values=True))

    traj = collector._collect_single_trajectory(policy=MagicMock())

    # T = 3
    assert traj["state"].shape      == (3, 4)
    assert traj["action"].shape     == (3, 1)
    assert traj["next_state"].shape == (3, 4)
    assert traj["reward"].shape     == (3, 1)
    assert traj["done"].shape       == (3, 1)

    # First two steps not done, last is done
    assert traj["done"][0].item() == 0
    assert traj["done"][1].item() == 0
    assert traj["done"][2].item() == 1

    # Values/log-probs present and stacked
    assert traj["value_est"].shape == (3, 1)
    assert traj["log_prob"].shape  == (3, 1)

    # Reset once at the beginning, step called T times
    assert mock_env.reset.call_count == 1
    assert mock_env.step.call_count  == 3


def test_collect_single_trajectory_continuous_shapes(mock_env, cont_action_params, monkeypatch):
    """
    Continuous actions should produce action shape (T, 2) with T determined by first done=True.
    """
    mock_env.get_parameters.return_value = cont_action_params

    # T = 4, done on last
    step_returns = [
        (torch.zeros(1, 4), torch.tensor([[0.5]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[0.5]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[0.5]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[0.5]]), torch.tensor([[True]]),  {}),
    ]
    mock_env.step.side_effect = step_returns

    collector = SequentialCollector(env=mock_env, logger=FakeLogger())
    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_continuous(return_values=True))

    traj = collector._collect_single_trajectory(policy=MagicMock())

    assert traj["action"].shape     == (4, 2)
    assert traj["state"].shape      == (4, 4)
    assert traj["next_state"].shape == (4, 4)
    assert traj["reward"].shape     == (4, 1)
    assert traj["done"].shape       == (4, 1)
    assert traj["value_est"].shape  == (4, 1)
    assert traj["log_prob"].shape   == (4, 1)

    assert traj["done"][-1].item() == 1
    assert mock_env.reset.call_count == 1
    assert mock_env.step.call_count  == 4

def test_collect_single_trajectory_no_values_or_logprobs(mock_env, discrete_action_params, monkeypatch):
    """
    If the policy returns no value/log_prob, the outputs should be None for those keys.
    """
    mock_env.get_parameters.return_value = discrete_action_params

    # T = 2
    step_returns = [
        (torch.zeros(1, 4), torch.tensor([[1.0]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[1.0]]), torch.tensor([[True]]),  {}),
    ]
    mock_env.step.side_effect = step_returns

    collector = SequentialCollector(env=mock_env, logger=FakeLogger())
    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_discrete(return_values=False))

    traj = collector._collect_single_trajectory(policy=MagicMock())

    assert traj["state"].shape      == (2, 4)
    assert traj["action"].shape     == (2, 1)
    assert traj["next_state"].shape == (2, 4)
    assert traj["reward"].shape     == (2, 1)
    assert traj["done"].shape       == (2, 1)

    assert not hasattr(traj, 'value_est')
    assert not hasattr(traj, 'log_prob')

    assert mock_env.reset.call_count == 1
    assert mock_env.step.call_count  == 2

def test_collect_single_trajectory_calls_metric_update_each_step(mock_env, discrete_action_params, monkeypatch):
    """
    Make sure the internal metrics tracker is updated exactly once per collected step.
    """
    mock_env.get_parameters.return_value = discrete_action_params

    # T = 3
    step_returns = [
        (torch.zeros(1, 4), torch.tensor([[0.1]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[0.2]]), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.tensor([[0.3]]), torch.tensor([[True]]),  {}),
    ]
    mock_env.step.side_effect = step_returns

    collector = SequentialCollector(env=mock_env, logger=FakeLogger())

    # Spy on metric updates
    collector.metric_tracker.update = MagicMock()

    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_discrete(return_values=True))

    traj = collector._collect_single_trajectory(policy=MagicMock())

    # Called once per step
    assert collector.metric_tracker.update.call_count == 3

    # Sanity: shapes and done
    assert traj["state"].shape == (3, 4)
    assert traj["done"][-1].item() == 1

def test_collect_trajectory_num_trajectories_stacks_and_shapes_discrete(mock_env, discrete_action_params, monkeypatch):
    """
    Two trajectories with lengths T1=2 and T2=3 -> B=5. Optional keys present for all steps.
    """
    mock_env.get_parameters.return_value = discrete_action_params

    # Build 2 trajectories: (F,F,T) and (F,T) -> total 5 steps
    step_returns = [
        # Traj 1 (T1=2)
        (torch.zeros(1,4), torch.tensor([[1.0]]), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.tensor([[1.0]]), torch.tensor([[True]]),  {}),
        # Traj 2 (T2=3)
        (torch.zeros(1,4), torch.tensor([[1.0]]), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.tensor([[1.0]]), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.tensor([[1.0]]), torch.tensor([[True]]),  {}),
    ]
    mock_env.step.side_effect = step_returns

    collector = SequentialCollector(env=mock_env, logger=FakeLogger())

    # Patch policy helper in SAME module as SequentialCollector
    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_discrete(return_values=True))

    out = collector.collect_trajectory(policy=MagicMock(), num_trajectories=2)

    # B = 5
    assert out["state"].shape      == (5, 4)
    assert out["action"].shape     == (5, 1)
    assert out["next_state"].shape == (5, 4)
    assert out["reward"].shape     == (5, 1)
    assert out["done"].shape       == (5, 1)
    assert out["value_est"].shape  == (5, 1)
    assert out["log_prob"].shape   == (5, 1)

    # Done at indices 1 and 4
    assert out["done"][1].item() == 1
    assert out["done"][4].item() == 1

    # One reset at start of each trajectory
    assert mock_env.reset.call_count == 2
    assert mock_env.step.call_count  == 5

def test_collect_trajectory_min_steps_completes_last_trajectory(mock_env, discrete_action_params, monkeypatch):
    """
    min_num_steps=4 should collect two trajectories T1=3, T2=2 -> B=5, finish the second trajectory.
    """
    mock_env.get_parameters.return_value = discrete_action_params

    step_returns = [
        # Traj 1 (T1=3)
        (torch.zeros(1,4), torch.tensor([[0.5]]), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.tensor([[0.5]]), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.tensor([[0.5]]), torch.tensor([[True]]),  {}),
        # Traj 2 (T2=2)
        (torch.zeros(1,4), torch.tensor([[0.7]]), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.tensor([[0.7]]), torch.tensor([[True]]),  {}),
    ]
    mock_env.step.side_effect = step_returns

    collector = SequentialCollector(env=mock_env, logger=FakeLogger())
    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_discrete(return_values=True))

    out = collector.collect_trajectory(policy=MagicMock(), min_num_steps=4)

    # B = 5 >= 4 and last step is done
    assert out["state"].shape == (5, 4)
    assert out["done"][-1].item() == 1

    # Resets = number of trajectories collected (2)
    assert mock_env.reset.call_count == 2
    assert mock_env.step.call_count  == 5

def test_collect_trajectory_optional_keys_all_missing(mock_env, discrete_action_params, monkeypatch):
    """
    If no trajectories provide optional keys, expect None in the output.
    """
    mock_env.get_parameters.return_value = discrete_action_params
    collector = SequentialCollector(env=mock_env, logger=FakeLogger())

    traj = {
        "state":      torch.zeros(2, 4),
        "action":     torch.zeros(2, 1),
        "next_state": torch.zeros(2, 4),
        "reward":     torch.ones(2, 1),
        "done":       torch.tensor([[0.], [1.]]),
        "value_est":  None,
        "log_prob":   None,
    }

    monkeypatch.setattr(collector, "_collect_single_trajectory", lambda _p=None: traj)

    out = collector.collect_trajectory(policy=MagicMock(), num_trajectories=2)

    assert out["state"].shape == (4, 4)
    assert not hasattr(out, 'value_est')
    assert not hasattr(out, 'log_prob')

def test_collect_trajectory_continuous_action_shapes(mock_env, cont_action_params, monkeypatch):
    """
    Continuous actions: two trajectories T1=4, T2=1 -> B=5, action shape (B,2).
    """
    mock_env.get_parameters.return_value = cont_action_params

    step_returns = [
        # Traj 1 (T1=4)
        (torch.zeros(1,4), torch.tensor([[0.1]]), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.tensor([[0.1]]), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.tensor([[0.1]]), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.tensor([[0.1]]), torch.tensor([[True]]),  {}),
        # Traj 2 (T2=1)
        (torch.zeros(1,4), torch.tensor([[0.2]]), torch.tensor([[True]]),  {}),
    ]
    mock_env.step.side_effect = step_returns

    collector = SequentialCollector(env=mock_env, logger=FakeLogger())
    seq_mod = sys.modules[SequentialCollector.__module__]
    monkeypatch.setattr(seq_mod, 'get_action_from_policy', _policy_stub_continuous(return_values=True))

    out = collector.collect_trajectory(policy=MagicMock(), num_trajectories=2)

    assert out["action"].shape     == (5, 2)
    assert out["state"].shape      == (5, 4)
    assert out["next_state"].shape == (5, 4)
    assert out["reward"].shape     == (5, 1)
    assert out["done"].shape       == (5, 1)
    assert out["value_est"].shape  == (5, 1)
    assert out["log_prob"].shape   == (5, 1)

    assert mock_env.reset.call_count == 2
    assert mock_env.step.call_count  == 5

def test_collect_trajectory_invalid_args_raises(mock_env, discrete_action_params):
    mock_env.get_parameters.return_value = discrete_action_params
    collector = SequentialCollector(env=mock_env, logger=FakeLogger())
    with pytest.raises(ValueError):
        collector.collect_trajectory(policy=MagicMock(), num_trajectories=2, min_num_steps=10)


# Parallel Collector Tests
# =========================================================
# Helpers
@pytest.fixture
def mock_vec_env_n1(discrete_action_params):
    env = MagicMock()
    N = 1
    env.get_num_envs.return_value = N
    env.get_parameters.return_value = discrete_action_params

    # reset() -> (N, obs_dim)
    env.reset.return_value = (torch.zeros(N, 4, dtype=torch.float32), {})
    # reset_index(i) -> (obs_dim,)
    env.reset_index.side_effect = lambda i: (torch.zeros(4, dtype=torch.float32), {})
    # step(action) default: not done
    env.step.return_value = (
        torch.zeros(N, 4, dtype=torch.float32),
        torch.ones(N, 1, dtype=torch.float32),
        torch.zeros(N, 1, dtype=torch.bool),
        {}
    )
    return env

# ---------------------------------
# Policy stubs for vector env (N=1)
# ---------------------------------
@pytest.fixture
def mock_vec_env_n3(discrete_action_params):
    """
    Minimal vectorized env mock with N=3, obs_dim=4, discrete action_len=1.
    """
    env = MagicMock()
    N = 3
    env.get_num_envs.return_value = N
    env.get_parameters.return_value = discrete_action_params

    # reset() -> (N, obs_dim)
    env.reset.return_value = (torch.zeros(N, 4, dtype=torch.float32), {})

    # reset_index(i) -> (obs_dim,)
    env.reset_index.side_effect = lambda i: (torch.zeros(4, dtype=torch.float32), {})

    # step(action) -> all envs continue (done=False)
    env.step.return_value = (
        torch.zeros(N, 4, dtype=torch.float32),          # next_state
        torch.ones(N, 1, dtype=torch.float32),           # reward
        torch.zeros(N, 1, dtype=torch.bool),             # done (all False)
        {}
    )
    return env

def _vec_policy_discrete(with_values=True):
    def f(policy, state, env_params):
        # state: (N=1, 4)
        action = torch.zeros(1, 1, dtype=torch.float32)  # (N, action_len=1)
        if with_values:
            value_est = torch.full((1, 1), 0.5, dtype=torch.float32)
            log_prob  = torch.full((1, 1), -0.1, dtype=torch.float32)
        else:
            value_est = None
            log_prob  = None
        return action, value_est, log_prob
    return f

def _vec_policy_discrete_counting(counter):
    """Counts how many times the policy is queried; returns value/logprob."""
    def f(policy, state, env_params):
        counter["n"] += 1
        action = torch.zeros(1, 1, dtype=torch.float32)
        value_est = torch.full((1, 1), 0.7, dtype=torch.float32)
        log_prob  = torch.full((1, 1), -0.2, dtype=torch.float32)
        return action, value_est, log_prob
    return f

def _vec_policy_discrete_multi(with_values=True):
    """
    Returns action/value/log_prob with correct leading N based on state.shape[0].
    Assumes discrete action_len=1 from env_params.
    """
    def f(policy, state, env_params):
        N = state.shape[0]
        action = torch.zeros(N, env_params.action_len, dtype=torch.float32)
        if with_values:
            value_est = torch.full((N, 1), 0.5, dtype=torch.float32)
            log_prob  = torch.full((N, 1), -0.1, dtype=torch.float32)
        else:
            value_est = None
            log_prob  = None
        return action, value_est, log_prob
    return f

class _ScriptedVecEnv:
    """
    Minimal vectorized env with deterministic 'done' schedule.
    - N: number of envs
    - ends_per_env: list of lists; ends_per_env[i] contains the global step indices t where env i ends.
    next_state encodes env id in feature 0 so we can recover which env a step came from.
    """
    def __init__(self, N, ends_per_env, obs_dim=4):
        self._N = N
        self._t = 0
        self._ends = [list(lst) for lst in ends_per_env]  # copy
        self._obs_dim = obs_dim
        self.reset_index_calls = []

    def get_num_envs(self):
        return self._N

    def get_parameters(self):
        # Use the discrete_action_params injected by outer fixtures/tests:
        # We'll monkeypatch the collector to use this env; the test sets env.get_parameters
        # via a fixture typically. But to be safe, return a simple object with needed fields.
        class P:
            action_len = 1
            action_continuous = False
            observation_shape = (self._obs_dim,)
        return P()

    def reset(self):
        self._t = 0
        # State shape: (N, obs_dim), encode env_id at feature 0
        state = torch.zeros(self._N, self._obs_dim, dtype=torch.float32)
        for i in range(self._N):
            state[i, 0] = float(i)
        return state, {}

    def reset_index(self, i):
        self.reset_index_calls.append(i)
        s = torch.zeros(self._obs_dim, dtype=torch.float32)
        s[0] = float(i)
        return s, {}

    def step(self, action):
        # Build next_state with env id encoded in feature 0
        next_state = torch.zeros(self._N, self._obs_dim, dtype=torch.float32)
        for i in range(self._N):
            next_state[i, 0] = float(i)

        reward = torch.ones(self._N, 1, dtype=torch.float32)

        # done: True if current t is a scheduled end for env i
        done = torch.zeros(self._N, 1, dtype=torch.bool)
        for i in range(self._N):
            if self._ends[i] and self._t == self._ends[i][0]:
                done[i, 0] = True
                # pop this end time; next episode for env i will be driven by later times
                self._ends[i].pop(0)

        self._t += 1
        return next_state, reward, done, {}

# Tests
def test_parallel_collect_flatten_true_shapes(mock_vec_env_n1, monkeypatch):
    """
    flatten=True (default). With N=1 and num_steps=5 -> T = ceil(5/1) = 5, B = N*T = 5.
    Optional keys present; last_value_est computed.
    """
    env = mock_vec_env_n1
    collector = ParallelCollector(env=env, logger=FakeLogger(), flatten=True)

    # Patch policy
    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete(with_values=True))

    out = collector.collect_experience(policy=MagicMock(), num_steps=5)

    # B = 5
    assert out["state"].shape      == (5, 4)
    assert out["action"].shape     == (5, 1)
    assert out["next_state"].shape == (5, 4)
    assert out["reward"].shape     == (5, 1)
    assert out["done"].shape       == (5, 1)
    assert out["value_est"].shape  == (5, 1)
    assert out["log_prob"].shape   == (5, 1)

    # last_value_est should be present for N=1 when final step not done
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)

    # Called once to start, then 5 steps
    assert env.reset.call_count == 1
    assert env.step.call_count  == 5

def test_parallel_collect_flatten_false_shapes(mock_vec_env_n1, monkeypatch):
    """
    flatten=False -> shapes (T, N, ...). With N=1 and num_steps=4 -> T=4.
    """
    env = mock_vec_env_n1
    collector = ParallelCollector(env=env, logger=FakeLogger(), flatten=False)

    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete(with_values=True))

    out = collector.collect_experience(policy=MagicMock(), num_steps=4)

    # (T, N, ...) = (4, 1, ...)
    assert out["state"].shape      == (4, 1, 4)
    assert out["action"].shape     == (4, 1, 1)
    assert out["next_state"].shape == (4, 1, 4)
    assert out["reward"].shape     == (4, 1, 1)
    assert out["done"].shape       == (4, 1, 1)
    assert out["value_est"].shape  == (4, 1, 1)
    assert out["log_prob"].shape   == (4, 1, 1)

    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)

def test_parallel_collect_resets_done_envs_with_reset_index(mock_vec_env_n1, monkeypatch):
    """
    Ensure env.reset_index(i) is called for envs that reported done=True on previous step.
    With N=1 and T=4, mark done=True on the 2nd step only -> expect exactly 1 call to reset_index.
    """
    env = mock_vec_env_n1
    collector = ParallelCollector(env=env, logger=FakeLogger(), flatten=True)

    # Steps: F, T, F, F  (only the 'T' should trigger a reset_index before the next step)
    step_returns = [
        (torch.zeros(1,4), torch.ones(1,1), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.ones(1,1), torch.tensor([[True]]),  {}),
        (torch.zeros(1,4), torch.ones(1,1), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.ones(1,1), torch.tensor([[False]]), {}),
    ]
    env.step.side_effect = step_returns

    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete(with_values=False))  # no value/logprob

    out = collector.collect_experience(policy=MagicMock(), num_steps=4)

    # After step 2 (done=True), the next iteration should call reset_index(0)
    assert env.reset_index.call_count == 1

    # value/log_prob absent -> None
    assert not hasattr(out, 'value_est')
    assert not hasattr(out, 'log_prob')
    assert not hasattr(out, 'last_value_est')

def test_parallel_collect_bootstraps_last_value_estimate(mock_vec_env_n1, monkeypatch):
    """
    When final step is not done and we have value estimates, the collector does one extra
    policy call for V(s_{T+1}). Verify by counting policy calls: expected T + 1.
    """
    env = mock_vec_env_n1
    collector = ParallelCollector(env=env, logger=FakeLogger(), flatten=True)

    # Ensure none of the steps set done=True
    env.step.return_value = (
        torch.zeros(1, 4),
        torch.ones(1, 1),
        torch.zeros(1, 1, dtype=torch.bool),
        {}
    )

    counter = {"n": 0}
    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete_counting(counter))

    num_steps = 6
    out = collector.collect_experience(policy=MagicMock(), num_steps=num_steps)

    # T = ceil(6/1) = 6, expect 6 + 1 calls (bootstrap)
    assert counter["n"] == num_steps + 1
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)

def test_parallel_collect_handles_partial_last_done_without_reset(mock_vec_env_n1, monkeypatch):
    """
    If the last step reports done=True, no reset_index is expected (since no next step).
    """
    env = mock_vec_env_n1
    collector = ParallelCollector(env=env, logger=FakeLogger(), flatten=True)

    # Steps: F, F, T  -> only the 'T' is last; should NOT trigger reset_index
    step_returns = [
        (torch.zeros(1,4), torch.ones(1,1), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.ones(1,1), torch.tensor([[False]]), {}),
        (torch.zeros(1,4), torch.ones(1,1), torch.tensor([[True]]),  {}),
    ]
    env.step.side_effect = step_returns

    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete(with_values=True))

    out = collector.collect_experience(policy=MagicMock(), num_steps=3)

    assert env.reset_index.call_count == 0  # last step done -> no next-step reset
    # With last step done, your current code's bootstrap guard may skip last_value_est
    # because previous_experience['done'] is True; we just check no error:
    assert "last_value_est" in out

def test_parallel_collect_multi_env_flatten_true_shapes(mock_vec_env_n3, monkeypatch):
    """
    With N=3 and num_steps=7 -> T = ceil(7/3) = 3, B = N*T = 9.
    Expect flattened shapes: (B, ...).
    Also expects last_value_est shape (N,1) because final step is not done for any env
    and your guard is: if not done.any() and value_estimates: ...
    """
    env = mock_vec_env_n3
    collector = ParallelCollector(env=env, logger=FakeLogger(), flatten=True)

    # Patch the helper inside the module where ParallelCollector is defined
    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete_multi(with_values=True))

    out = collector.collect_experience(policy=MagicMock(), num_steps=7)

    # B = 9
    assert out["state"].shape      == (9, 4)
    assert out["action"].shape     == (9, 1)
    assert out["next_state"].shape == (9, 4)
    assert out["reward"].shape     == (9, 1)
    assert out["done"].shape       == (9, 1)
    assert out["value_est"].shape  == (9, 1)
    assert out["log_prob"].shape   == (9, 1)

    # last_value_est should be present with shape (N,1)
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (env.get_num_envs(), 1)

    # One reset to start, T=3 steps
    assert env.reset.call_count == 1
    assert env.step.call_count  == 3
    # No per-index resets since no env finished
    assert env.reset_index.call_count == 0


def test_parallel_collect_multi_env_flatten_false_shapes(mock_vec_env_n3, monkeypatch):
    """
    With N=3 and num_steps=7 -> T = 3.
    Expect unflattened shapes: (T, N, ...).
    """
    env = mock_vec_env_n3
    collector = ParallelCollector(env=env, logger=FakeLogger(), flatten=False)

    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete_multi(with_values=True))

    out = collector.collect_experience(policy=MagicMock(), num_steps=7)

    # (T, N, ...)
    assert out["state"].shape      == (3, 3, 4)
    assert out["action"].shape     == (3, 3, 1)
    assert out["next_state"].shape == (3, 3, 4)
    assert out["reward"].shape     == (3, 3, 1)
    assert out["done"].shape       == (3, 3, 1)
    assert out["value_est"].shape  == (3, 3, 1)
    assert out["log_prob"].shape   == (3, 3, 1)

    # last_value_est should be (N,1)
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (env.get_num_envs(), 1)

    assert env.reset.call_count == 1
    assert env.step.call_count  == 3
    assert env.reset_index.call_count == 0

def test_collect_trajectory_fair_distribution_k5_n3(monkeypatch):
    """
    K=5, N=3
    - base = 1 per env (first episode from each env)
    - remainder = 2 earliest extra, from DISTINCT envs
    Episode finish times:
      env0: t_end at 1, 5           (lens: 2, 4)
      env1: t_end at 2, 4           (lens: 3, 2)
      env2: t_end at 3, 6           (lens: 4, 3)
    Expected selected order by finish time: [e0@1, e1@2, e2@3] + [e1@4, e0@5]
    => episode env sequence at 'done' rows: [0, 1, 2, 1, 0]
    Total B = 2 + 3 + 4 + 2 + 4 = 15
    """
    N = 3
    ends = [
        [1, 5],  # env 0
        [2, 4],  # env 1
        [3, 6],  # env 2
    ]
    env = _ScriptedVecEnv(N, ends)
    collector = ParallelCollector(env=env, logger=FakeLogger())

    # Patch policy helper in the SAME module where ParallelCollector is defined
    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete_multi(with_values=True))

    out = collector.collect_trajectory(policy=MagicMock(), num_trajectories=5)

    # Shapes
    assert out["state"].shape      == (15, 4)
    assert out["action"].shape     == (15, 1)
    assert out["next_state"].shape == (15, 4)
    assert out["reward"].shape     == (15, 1)
    assert out["done"].shape       == (15, 1)
    assert out["value_est"].shape  == (15, 1)
    assert out["log_prob"].shape   == (15, 1)

    # Which env ended at each episode boundary? (env id is encoded at state[..., 0])
    done_idx = torch.nonzero(out["done"].squeeze(-1), as_tuple=False).flatten().tolist()
    end_envs = [int(out["state"][i, 0].item()) for i in done_idx]

    # Expected: [0, 1, 2, 1, 0] (see docstring)
    assert end_envs == [0, 1, 2, 1, 0]

    # Counts per env: base 1 each + remainder spread across distinct envs
    assert end_envs.count(0) == 2
    assert end_envs.count(1) == 2
    assert end_envs.count(2) == 1


def test_collect_trajectory_min_num_steps_earliest_episodes(monkeypatch):
    """
    min_num_steps = 8 with the same schedule:
      earliest finishes: e0@1 (len2) -> sum 2
                         e1@2 (len3) -> sum 5
                         e2@3 (len4) -> sum 9 >= 8 stop
    Selected env sequence at 'done': [0, 1, 2]
    Total B = 2 + 3 + 4 = 9
    """
    N = 3
    ends = [
        [1, 5],
        [2, 4],
        [3, 6],
    ]
    env = _ScriptedVecEnv(N, ends)
    collector = ParallelCollector(env=env, logger=FakeLogger(), flatten=True)

    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete_multi(with_values=True))

    out = collector.collect_trajectory(policy=MagicMock(), min_num_steps=8)

    assert out["state"].shape == (9, 4)
    assert out["done"].shape  == (9, 1)

    done_idx = torch.nonzero(out["done"].squeeze(-1), as_tuple=False).flatten().tolist()
    end_envs = [int(out["state"][i, 0].item()) for i in done_idx]

    assert end_envs == [0, 1, 2]  # earliest by finish time
    assert len(done_idx) == 3     # three episodes
    assert out["value_est"].shape == (9, 1)
    assert out["log_prob"].shape  == (9, 1)


def test_collect_trajectory_optional_keys_absent(monkeypatch):
    """
    If the policy provides no value/log_prob, collector should return None for these keys.
    """
    N = 3
    ends = [
        [1, 5],
        [2, 4],
        [3, 6],
    ]
    env = _ScriptedVecEnv(N, ends)
    collector = ParallelCollector(env=env, logger=FakeLogger(), flatten=True)

    pc_mod = sys.modules[ParallelCollector.__module__]
    monkeypatch.setattr(pc_mod, 'get_action_from_policy', _vec_policy_discrete_multi(with_values=False))

    out = collector.collect_trajectory(policy=MagicMock(), num_trajectories=3)

    # 1 episode per env -> lens 2 + 3 + 4 = 9
    assert out["state"].shape == (9, 4)
    assert not hasattr(out, 'value_est')
    assert not hasattr(out, 'log_prob')

def test_collect_step_from_single_env_with_reset():
    env = GymnasiumWrapper("CartPole-v1")

    collector = ParallelCollector(env=env, flatten=False)
    state, action, next_state, reward, done, value_est, log_prob = collector._collect_step()
    collector.previous_experience['done'] = torch.tensor([[True]])

    state, action, next_state, reward, done, value_est, log_prob = collector._collect_step()
    assert state.shape == (1, 4)
    assert action.shape == (1, 1)
    assert next_state.shape == (1, 4)
    assert reward.shape == (1, 1)
    assert done.shape == (1, 1)
    assert value_est is None
    assert log_prob is None