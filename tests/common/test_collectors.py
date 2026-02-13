import pytest
import torch
from unittest.mock import MagicMock
from prt_rl.env.wrappers import GymnasiumWrapper
from prt_rl.env.interface import EnvParams
from prt_rl.common.collectors import Collector, MetricsTracker


class FakeLogger:
    def __init__(self):
        self.scalars = []  # list of (name, value, iteration)

    def log_scalar(self, name: str, value: float, iteration: int = None):
        self.scalars.append((name, float(value), int(iteration) if iteration is not None else None))

    def _by_name(self, name):
        """Return list of (value, iteration) for a metric name."""
        return [(v, it) for (n, v, it) in self.scalars if n == name]
    
    def should_log(self, iteration):
        return True


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
        (torch.tensor(5.0), torch.tensor([5.0])),  # Single scalar reward -> (1,)
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])),  # reward (N,) -> (N,)
        (torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([1.0, 2.0, 3.0])),  # reward (N, 1) -> (N,)
        (torch.ones((3, 4, 1)), torch.tensor([4.0, 4.0, 4.0])),  # reward (N, 4, 1) -> (N,)
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
    assert tr.cumulative_reward == pytest.approx(1 + 2 + 3 + 4)
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
    tr.update(torch.tensor([[2.0], [20.0], [200.0]]), torch.tensor([[True], [False], [True]]))

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
    r1 = torch.tensor([[[1.0], [0.5], [0.5]], [[2.0], [1.0], [1.0]]])  # sums: [2.0, 4.0]
    d1 = torch.tensor([[False], [False]])
    tr.update(r1, d1)

    # step 2, both end
    r2 = torch.tensor([[[0.0], [1.0], [1.0]], [[1.0], [1.0], [0.0]]])  # sums: [2.0, 2.0]
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


# Collector Tests
# =========================================================
# Mock policy that returns action and policy_vals dictionary
class MockPolicy:
    """Mock policy that returns action and policy_vals dictionary."""

    def __init__(self, action_shape, with_values=True, value_val=0.5, log_prob_val=-0.1):
        self.action_shape = action_shape
        self.with_values = with_values
        self.value_val = value_val
        self.log_prob_val = log_prob_val
        self.call_count = 0

    def act(self, state, deterministic=False):
        self.call_count += 1
        N = state.shape[0]
        action = torch.zeros(N, *self.action_shape, dtype=torch.float32)
        policy_vals = {}
        if self.with_values:
            policy_vals["value"] = torch.full((N, 1), self.value_val, dtype=torch.float32)
            policy_vals["log_prob"] = torch.full((N, 1), self.log_prob_val, dtype=torch.float32)
        return action, policy_vals


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
        {},
    )
    return env


@pytest.fixture
def mock_vec_env_n3(discrete_action_params):
    """Minimal vectorized env mock with N=3, obs_dim=4, discrete action_len=1."""
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
        torch.zeros(N, 4, dtype=torch.float32),  # next_state
        torch.ones(N, 1, dtype=torch.float32),  # reward
        torch.zeros(N, 1, dtype=torch.bool),  # done (all False)
        {},
    )
    return env


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


# Tests for collect_experience
def test_collect_flatten_true_shapes(mock_vec_env_n1):
    """
    flatten=True (default). With N=1 and num_steps=5 -> T = ceil(5/1) = 5, B = N*T = 5.
    Optional keys present; last_value_est computed.
    """
    env = mock_vec_env_n1
    collector = Collector(env=env, logger=FakeLogger(), flatten=True)
    policy = MockPolicy(action_shape=(1,), with_values=True)

    out = collector.collect_experience(policy=policy, num_steps=5)

    # B = 5
    assert out["state"].shape == (5, 4)
    assert out["action"].shape == (5, 1)
    assert out["next_state"].shape == (5, 4)
    assert out["reward"].shape == (5, 1)
    assert out["done"].shape == (5, 1)
    assert out["value"].shape == (5, 1)
    assert out["log_prob"].shape == (5, 1)

    # last_value_est should be present for N=1 when final step not done
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)

    # Called once to start, then 5 steps
    assert env.reset.call_count == 1
    assert env.step.call_count == 5


def test_collect_flatten_false_shapes(mock_vec_env_n1):
    """
    flatten=False -> shapes (T, N, ...). With N=1 and num_steps=4 -> T=4.
    """
    env = mock_vec_env_n1
    collector = Collector(env=env, logger=FakeLogger(), flatten=False)
    policy = MockPolicy(action_shape=(1,), with_values=True)

    out = collector.collect_experience(policy=policy, num_steps=4)

    # (T, N, ...) = (4, 1, ...)
    assert out["state"].shape == (4, 1, 4)
    assert out["action"].shape == (4, 1, 1)
    assert out["next_state"].shape == (4, 1, 4)
    assert out["reward"].shape == (4, 1, 1)
    assert out["done"].shape == (4, 1, 1)
    assert out["value"].shape == (4, 1, 1)
    assert out["log_prob"].shape == (4, 1, 1)

    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)


def test_collect_resets_done_envs_with_reset_index(mock_vec_env_n1):
    """
    Ensure env.reset_index(i) is called for envs that reported done=True on previous step.
    With N=1 and T=4, mark done=True on the 2nd step only -> expect exactly 1 call to reset_index.
    """
    env = mock_vec_env_n1
    collector = Collector(env=env, logger=FakeLogger(), flatten=True)

    # Steps: F, T, F, F  (only the 'T' should trigger a reset_index before the next step)
    step_returns = [
        (torch.zeros(1, 4), torch.ones(1, 1), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.ones(1, 1), torch.tensor([[True]]), {}),
        (torch.zeros(1, 4), torch.ones(1, 1), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.ones(1, 1), torch.tensor([[False]]), {}),
    ]
    env.step.side_effect = step_returns

    policy = MockPolicy(action_shape=(1,), with_values=False)
    out = collector.collect_experience(policy=policy, num_steps=4)

    # After step 2 (done=True), the next iteration should call reset_index(0)
    assert env.reset_index.call_count == 1

    # value/log_prob absent -> not in output
    assert "value" not in out
    assert "log_prob" not in out
    assert "last_value_est" not in out


def test_collect_bootstraps_last_value_estimate(mock_vec_env_n1):
    """
    When final step is not done and we have value estimates, the collector does one extra
    policy call for V(s_{T+1}). Verify by counting policy calls: expected T + 1.
    """
    env = mock_vec_env_n1
    collector = Collector(env=env, logger=FakeLogger(), flatten=True)

    # Ensure none of the steps set done=True
    env.step.return_value = (
        torch.zeros(1, 4),
        torch.ones(1, 1),
        torch.zeros(1, 1, dtype=torch.bool),
        {},
    )

    policy = MockPolicy(action_shape=(1,), with_values=True)
    num_steps = 6
    out = collector.collect_experience(policy=policy, num_steps=num_steps)

    # T = ceil(6/1) = 6, expect 6 + 1 calls (bootstrap)
    assert policy.call_count == num_steps + 1
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (1, 1)


def test_collect_handles_partial_last_done_without_reset(mock_vec_env_n1):
    """
    If the last step reports done=True, no reset_index is expected (since no next step).
    """
    env = mock_vec_env_n1
    collector = Collector(env=env, logger=FakeLogger(), flatten=True)

    # Steps: F, F, T  -> only the 'T' is last; should NOT trigger reset_index
    step_returns = [
        (torch.zeros(1, 4), torch.ones(1, 1), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.ones(1, 1), torch.tensor([[False]]), {}),
        (torch.zeros(1, 4), torch.ones(1, 1), torch.tensor([[True]]), {}),
    ]
    env.step.side_effect = step_returns

    policy = MockPolicy(action_shape=(1,), with_values=True)
    out = collector.collect_experience(policy=policy, num_steps=3)

    assert env.reset_index.call_count == 0  # last step done -> no next-step reset
    # With last step done, bootstrap guard may skip last_value_est
    assert "last_value_est" in out


def test_collect_multi_env_flatten_true_shapes(mock_vec_env_n3):
    """
    With N=3 and num_steps=7 -> T = ceil(7/3) = 3, B = N*T = 9.
    Expect flattened shapes: (B, ...).
    """
    env = mock_vec_env_n3
    collector = Collector(env=env, logger=FakeLogger(), flatten=True)
    policy = MockPolicy(action_shape=(1,), with_values=True)

    out = collector.collect_experience(policy=policy, num_steps=7)

    # B = 9
    assert out["state"].shape == (9, 4)
    assert out["action"].shape == (9, 1)
    assert out["next_state"].shape == (9, 4)
    assert out["reward"].shape == (9, 1)
    assert out["done"].shape == (9, 1)
    assert out["value"].shape == (9, 1)
    assert out["log_prob"].shape == (9, 1)

    # last_value_est should be present with shape (N,1)
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (env.get_num_envs(), 1)

    # One reset to start, T=3 steps
    assert env.reset.call_count == 1
    assert env.step.call_count == 3
    # No per-index resets since no env finished
    assert env.reset_index.call_count == 0


def test_collect_multi_env_flatten_false_shapes(mock_vec_env_n3):
    """
    With N=3 and num_steps=7 -> T = 3.
    Expect unflattened shapes: (T, N, ...).
    """
    env = mock_vec_env_n3
    collector = Collector(env=env, logger=FakeLogger(), flatten=False)
    policy = MockPolicy(action_shape=(1,), with_values=True)

    out = collector.collect_experience(policy=policy, num_steps=7)

    # (T, N, ...)
    assert out["state"].shape == (3, 3, 4)
    assert out["action"].shape == (3, 3, 1)
    assert out["next_state"].shape == (3, 3, 4)
    assert out["reward"].shape == (3, 3, 1)
    assert out["done"].shape == (3, 3, 1)
    assert out["value"].shape == (3, 3, 1)
    assert out["log_prob"].shape == (3, 3, 1)

    # last_value_est should be (N,1)
    assert out["last_value_est"] is not None
    assert out["last_value_est"].shape == (env.get_num_envs(), 1)

    assert env.reset.call_count == 1
    assert env.step.call_count == 3
    assert env.reset_index.call_count == 0


def test_collect_experience_with_custom_policy_vals():
    """
    Test that custom keys in policy_vals are properly collected and stacked.
    """

    class CustomPolicy:
        def act(self, state, deterministic=False):
            N = state.shape[0]
            action = torch.zeros(N, 1)
            policy_vals = {
                "value": torch.full((N, 1), 0.5),
                "log_prob": torch.full((N, 1), -0.1),
                "entropy": torch.full((N, 1), 0.3),
                "custom_key": torch.full((N, 2), 1.0),
            }
            return action, policy_vals

    env = MagicMock()
    env.get_num_envs.return_value = 2
    env.get_parameters.return_value = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=1,
        observation_shape=(4,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )
    env.reset.return_value = (torch.zeros(2, 4), {})
    env.step.return_value = (torch.zeros(2, 4), torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.bool), {})

    collector = Collector(env=env, logger=FakeLogger(), flatten=True)
    out = collector.collect_experience(policy=CustomPolicy(), num_steps=4)

    # N=2, num_steps=4 -> T=2, B=4
    assert out["value"].shape == (4, 1)
    assert out["log_prob"].shape == (4, 1)
    assert out["entropy"].shape == (4, 1)
    assert out["custom_key"].shape == (4, 2)


# Tests for collect_trajectory
def test_collect_trajectory_fair_distribution_k5_n3():
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
    collector = Collector(env=env, logger=FakeLogger())
    policy = MockPolicy(action_shape=(1,), with_values=True)

    out = collector.collect_trajectory(policy=policy, num_trajectories=5)

    # Shapes
    assert out["state"].shape == (15, 4)
    assert out["action"].shape == (15, 1)
    assert out["next_state"].shape == (15, 4)
    assert out["reward"].shape == (15, 1)
    assert out["done"].shape == (15, 1)
    assert out["value"].shape == (15, 1)
    assert out["log_prob"].shape == (15, 1)

    # Which env ended at each episode boundary? (env id is encoded at state[..., 0])
    done_idx = torch.nonzero(out["done"].squeeze(-1), as_tuple=False).flatten().tolist()
    end_envs = [int(out["state"][i, 0].item()) for i in done_idx]

    # Expected: [0, 1, 2, 1, 0] (see docstring)
    assert end_envs == [0, 1, 2, 1, 0]

    # Counts per env: base 1 each + remainder spread across distinct envs
    assert end_envs.count(0) == 2
    assert end_envs.count(1) == 2
    assert end_envs.count(2) == 1


def test_collect_trajectory_min_num_steps_earliest_episodes():
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
    collector = Collector(env=env, logger=FakeLogger(), flatten=True)
    policy = MockPolicy(action_shape=(1,), with_values=True)

    out = collector.collect_trajectory(policy=policy, min_num_steps=8)

    assert out["state"].shape == (9, 4)
    assert out["done"].shape == (9, 1)

    done_idx = torch.nonzero(out["done"].squeeze(-1), as_tuple=False).flatten().tolist()
    end_envs = [int(out["state"][i, 0].item()) for i in done_idx]

    assert end_envs == [0, 1, 2]  # earliest by finish time
    assert len(done_idx) == 3  # three episodes
    assert out["value"].shape == (9, 1)
    assert out["log_prob"].shape == (9, 1)


def test_collect_trajectory_optional_keys_absent():
    """
    If the policy provides no value/log_prob, collector should not include these keys.
    """
    N = 3
    ends = [
        [1, 5],
        [2, 4],
        [3, 6],
    ]
    env = _ScriptedVecEnv(N, ends)
    collector = Collector(env=env, logger=FakeLogger(), flatten=True)
    policy = MockPolicy(action_shape=(1,), with_values=False)

    out = collector.collect_trajectory(policy=policy, num_trajectories=3)

    # 1 episode per env -> lens 2 + 3 + 4 = 9
    assert out["state"].shape == (9, 4)
    assert "value" not in out
    assert "log_prob" not in out


def test_collect_trajectory_with_custom_policy_vals():
    """
    Test that custom keys in policy_vals are properly collected in trajectories.
    """

    class CustomPolicy:
        def act(self, state, deterministic=False):
            N = state.shape[0]
            action = torch.zeros(N, 1)
            policy_vals = {
                "value": torch.full((N, 1), 0.5),
                "entropy": torch.full((N, 1), 0.3),
            }
            return action, policy_vals

    N = 2
    ends = [[1], [2]]  # env 0 ends at t=1, env 1 ends at t=2
    env = _ScriptedVecEnv(N, ends)
    collector = Collector(env=env, logger=FakeLogger(), flatten=True)

    out = collector.collect_trajectory(policy=CustomPolicy(), num_trajectories=2)

    # env 0: len 2, env 1: len 3 -> total 5
    assert out["state"].shape == (5, 4)
    assert out["value"].shape == (5, 1)
    assert out["entropy"].shape == (5, 1)


def test_collect_step_from_single_env_with_reset():
    """Test _collect_step with a real environment."""
    env = GymnasiumWrapper("CartPole-v1")
    collector = Collector(env=env, flatten=False)

    # Use a policy that returns integer actions for CartPole
    class CartPolePolicy:
        def act(self, state, deterministic=False):
            N = state.shape[0]
            action = torch.zeros(N, 1, dtype=torch.long)  # Integer action for CartPole
            return action, {}

    policy = CartPolePolicy()

    state, action, next_state, reward, done, policy_vals = collector._collect_step(policy)
    collector.previous_experience["done"] = torch.tensor([[True]])

    state, action, next_state, reward, done, policy_vals = collector._collect_step(policy)
    assert state.shape == (1, 4)
    assert action.shape == (1, 1)
    assert next_state.shape == (1, 4)
    assert reward.shape == (1, 1)
    assert done.shape == (1, 1)
    assert policy_vals == {}


def test_get_metric_tracker():
    """Test that get_metric_tracker returns the internal MetricsTracker."""
    env = MagicMock()
    env.get_num_envs.return_value = 1
    env.get_parameters.return_value = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=1,
        observation_shape=(4,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )

    collector = Collector(env=env, logger=FakeLogger())
    tracker = collector.get_metric_tracker()

    assert isinstance(tracker, MetricsTracker)
    assert tracker is collector.metric


# Tests for private helper methods
# =========================================================
class TestCollectorHelperMethods:
    """Tests for the refactored private helper methods of Collector."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple mock environment."""
        env = MagicMock()
        env.get_num_envs.return_value = 2
        env.get_parameters.return_value = EnvParams(
            action_len=1,
            action_continuous=False,
            action_min=0,
            action_max=1,
            observation_shape=(4,),
            observation_continuous=True,
            observation_min=-1.0,
            observation_max=1.0,
        )
        env.reset.return_value = (torch.zeros(2, 4), {})
        env.step.return_value = (torch.zeros(2, 4), torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.bool), {})
        return env

    def test_accumulate_policy_vals_new_key(self, simple_env):
        """Test _accumulate_policy_vals adds new keys correctly."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        accumulated = {}
        new_vals = {'value': torch.tensor([[1.0], [2.0]]), 'log_prob': torch.tensor([[-0.1], [-0.2]])}
        
        collector._accumulate_policy_vals(accumulated, new_vals)
        
        assert 'value' in accumulated
        assert 'log_prob' in accumulated
        assert len(accumulated['value']) == 1
        assert len(accumulated['log_prob']) == 1

    def test_accumulate_policy_vals_existing_key(self, simple_env):
        """Test _accumulate_policy_vals appends to existing keys."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        accumulated = {'value': [torch.tensor([[1.0]])]}
        new_vals = {'value': torch.tensor([[2.0]])}
        
        collector._accumulate_policy_vals(accumulated, new_vals)
        
        assert len(accumulated['value']) == 2

    def test_combine_step_data_flatten_true(self, simple_env):
        """Test _combine_step_data with flatten=True concatenates correctly."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        step_data = {
            'state': [torch.zeros(2, 4), torch.ones(2, 4)],
            'action': [torch.zeros(2, 1), torch.ones(2, 1)],
            'next_state': [torch.zeros(2, 4), torch.ones(2, 4)],
            'reward': [torch.zeros(2, 1), torch.ones(2, 1)],
            'done': [torch.zeros(2, 1), torch.ones(2, 1)],
            'policy_vals': {'value': [torch.zeros(2, 1), torch.ones(2, 1)]}
        }
        
        result = collector._combine_step_data(step_data, flatten=True)
        
        assert result['state'].shape == (4, 4)  # 2 envs * 2 steps
        assert result['action'].shape == (4, 1)
        assert result['value'].shape == (4, 1)

    def test_combine_step_data_flatten_false(self, simple_env):
        """Test _combine_step_data with flatten=False stacks correctly."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        step_data = {
            'state': [torch.zeros(2, 4), torch.ones(2, 4)],
            'action': [torch.zeros(2, 1), torch.ones(2, 1)],
            'next_state': [torch.zeros(2, 4), torch.ones(2, 4)],
            'reward': [torch.zeros(2, 1), torch.ones(2, 1)],
            'done': [torch.zeros(2, 1), torch.ones(2, 1)],
            'policy_vals': {'value': [torch.zeros(2, 1), torch.ones(2, 1)]}
        }
        
        result = collector._combine_step_data(step_data, flatten=False)
        
        assert result['state'].shape == (2, 2, 4)  # (T=2, N=2, 4)
        assert result['action'].shape == (2, 2, 1)
        assert result['value'].shape == (2, 2, 1)

    def test_stack_step_data(self, simple_env):
        """Test _stack_step_data creates correct tensor shapes."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        step_data = {
            'state': [torch.zeros(2, 4), torch.ones(2, 4), torch.full((2, 4), 2.0)],
            'action': [torch.zeros(2, 1), torch.ones(2, 1), torch.full((2, 1), 2.0)],
            'next_state': [torch.zeros(2, 4), torch.ones(2, 4), torch.full((2, 4), 2.0)],
            'reward': [torch.zeros(2, 1), torch.ones(2, 1), torch.full((2, 1), 2.0)],
            'done': [torch.zeros(2, 1), torch.ones(2, 1), torch.full((2, 1), 1.0)],
            'policy_vals': {'entropy': [torch.zeros(2, 1), torch.ones(2, 1), torch.full((2, 1), 2.0)]}
        }
        
        result = collector._stack_step_data(step_data)
        
        assert result['state'].shape == (3, 2, 4)  # (T=3, N=2, 4)
        assert result['action'].shape == (3, 2, 1)
        assert result['policy_vals']['entropy'].shape == (3, 2, 1)

    def test_find_episode_segments_single_env(self, simple_env):
        """Test _find_episode_segments with single environment."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        # Simulates: env 0 ends at t=2 and t=4, env 1 ends at t=3
        done_tensor = torch.tensor([
            [[False], [False]],  # t=0
            [[False], [False]],  # t=1
            [[True], [False]],   # t=2: env 0 done
            [[False], [True]],   # t=3: env 1 done
            [[True], [False]],   # t=4: env 0 done again
        ])
        
        result = collector._find_episode_segments(done_tensor, num_envs=2)
        
        assert len(result['per_env'][0]) == 2  # env 0 has 2 episodes
        assert len(result['per_env'][1]) == 1  # env 1 has 1 episode
        assert len(result['global']) == 3  # 3 total episodes
        
        # Check segment values for env 0
        assert result['per_env'][0][0] == (0, 0, 2)  # First episode: steps 0-2
        assert result['per_env'][0][1] == (0, 3, 4)  # Second episode: steps 3-4

    def test_find_episode_segments_no_completions(self, simple_env):
        """Test _find_episode_segments when no episodes complete."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        done_tensor = torch.tensor([
            [[False], [False]],
            [[False], [False]],
            [[False], [False]],
        ])
        
        result = collector._find_episode_segments(done_tensor, num_envs=2)
        
        assert len(result['per_env'][0]) == 0
        assert len(result['per_env'][1]) == 0
        assert len(result['global']) == 0

    def test_select_segments_by_trajectory_count_even_split(self, simple_env):
        """Test _select_segments_by_trajectory_count with even distribution."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        per_env_segments = [
            [(0, 0, 1), (0, 2, 3)],  # env 0: 2 episodes
            [(1, 0, 2), (1, 3, 4)],  # env 1: 2 episodes
        ]
        
        # Request 2 trajectories from 2 envs -> 1 each
        result = collector._select_segments_by_trajectory_count(per_env_segments, num_envs=2, num_trajectories=2)
        
        assert len(result) == 2
        # Should get one from each env
        env_indices = [seg[0] for seg in result]
        assert 0 in env_indices
        assert 1 in env_indices

    def test_select_segments_by_trajectory_count_with_remainder(self, simple_env):
        """Test _select_segments_by_trajectory_count with remainder episodes."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        per_env_segments = [
            [(0, 0, 1), (0, 2, 4)],  # env 0: ends at t=1, t=4
            [(1, 0, 2), (1, 3, 5)],  # env 1: ends at t=2, t=5
        ]
        
        # Request 3 trajectories from 2 envs -> 1 each + 1 extra (earliest)
        result = collector._select_segments_by_trajectory_count(per_env_segments, num_envs=2, num_trajectories=3)
        
        assert len(result) == 3
        # Extra should come from env 0 (ends at t=4, earlier than env 1's t=5)
        env_indices = [seg[0] for seg in result]
        assert env_indices.count(0) == 2
        assert env_indices.count(1) == 1

    def test_select_segments_by_step_count(self, simple_env):
        """Test _select_segments_by_step_count selects earliest episodes."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        # Segments: (env, start, end) -> length = end - start + 1
        global_segments = [
            (0, 0, 1),  # len=2, ends at t=1
            (1, 0, 3),  # len=4, ends at t=3
            (0, 2, 4),  # len=3, ends at t=4
        ]
        
        # Need at least 5 steps
        result = collector._select_segments_by_step_count(global_segments, min_num_steps=5)
        
        # First episode (len=2) + second (len=4) = 6 >= 5
        assert len(result) == 2
        assert result[0] == (0, 0, 1)  # First by end time
        assert result[1] == (1, 0, 3)  # Second by end time

    def test_extract_trajectory_segments(self, simple_env):
        """Test _extract_trajectory_segments extracts and concatenates correctly."""
        collector = Collector(env=simple_env, logger=FakeLogger())
        
        stacked_data = {
            'state': torch.arange(20).reshape(5, 2, 2).float(),  # (T=5, N=2, state_dim=2)
            'action': torch.arange(10).reshape(5, 2, 1).float(),
            'next_state': torch.arange(20).reshape(5, 2, 2).float(),
            'reward': torch.ones(5, 2, 1),
            'done': torch.zeros(5, 2, 1, dtype=torch.bool),
            'policy_vals': {'value': torch.arange(10).reshape(5, 2, 1).float()}
        }
        
        # Mark done at appropriate times
        stacked_data['done'][2, 0, 0] = True  # env 0 done at t=2
        stacked_data['done'][3, 1, 0] = True  # env 1 done at t=3
        
        selected_segments = [
            (0, 0, 2),  # env 0, steps 0-2 (length 3)
            (1, 0, 3),  # env 1, steps 0-3 (length 4)
        ]
        
        result = collector._extract_trajectory_segments(stacked_data, selected_segments)
        
        assert result['state'].shape == (7, 2)  # 3 + 4 = 7 steps
        assert result['action'].shape == (7, 1)
        assert result['value'].shape == (7, 1)

    def test_have_enough_trajectories_exact_match(self, simple_env):
        """Test _have_enough_trajectories with exact episode counts."""
        # 4 trajectories from 2 envs -> need 2 each
        assert Collector._have_enough_trajectories(4, 2, [2, 2]) is True
        assert Collector._have_enough_trajectories(4, 2, [2, 1]) is False
        assert Collector._have_enough_trajectories(4, 2, [3, 2]) is True

    def test_have_enough_trajectories_with_remainder(self, simple_env):
        """Test _have_enough_trajectories with remainder episodes."""
        # 5 trajectories from 2 envs -> need 2 each + 1 extra
        assert Collector._have_enough_trajectories(5, 2, [2, 2]) is False  # No extras
        assert Collector._have_enough_trajectories(5, 2, [3, 2]) is True   # One extra
        assert Collector._have_enough_trajectories(5, 2, [2, 3]) is True   # One extra

    def test_have_enough_trajectories_three_envs(self, simple_env):
        """Test _have_enough_trajectories with three environments."""
        # 7 trajectories from 3 envs -> need 2 each + 1 extra
        assert Collector._have_enough_trajectories(7, 3, [2, 2, 2]) is False
        assert Collector._have_enough_trajectories(7, 3, [3, 2, 2]) is True
        assert Collector._have_enough_trajectories(7, 3, [2, 3, 2]) is True


def test_collect_steps_helper():
    """Test the _collect_steps helper collects the correct number of steps."""
    env = MagicMock()
    env.get_num_envs.return_value = 2
    env.get_parameters.return_value = EnvParams(
        action_len=1,
        action_continuous=False,
        action_min=0,
        action_max=1,
        observation_shape=(4,),
        observation_continuous=True,
        observation_min=-1.0,
        observation_max=1.0,
    )
    env.reset.return_value = (torch.zeros(2, 4), {})
    env.step.return_value = (torch.zeros(2, 4), torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.bool), {})
    
    collector = Collector(env=env, logger=FakeLogger())
    policy = MockPolicy(action_shape=(1,), with_values=True)
    
    step_data = collector._collect_steps(policy, num_timesteps=5)
    
    assert len(step_data['state']) == 5
    assert len(step_data['action']) == 5
    assert len(step_data['policy_vals']['value']) == 5

