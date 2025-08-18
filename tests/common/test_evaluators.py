import copy
import math
import numpy as np
import pytest
import torch
import prt_rl.common.evaluators as MODULE
from prt_rl.env.wrappers import GymnasiumWrapper

# ---------------------------
# Fixtures
# ---------------------------
class FakeEnv:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs

class FakeLogger:
    def __init__(self):
        self.scalars = []  # (name, value, iteration)
        self.saved = []

    def log_scalar(self, name, value, iteration=None):
        self.scalars.append((name, float(value), iteration))

    def save_agent(self, agent, path):
        # Save a shallow copy to record what was passed
        self.saved.append((copy.deepcopy(agent), path))

class FakeCollector:
    """
    Mimics ParallelCollector.collect_trajectory by returning a batched dict:
      - 'reward': (B, 1)
      - 'done'  : (B, 1) with all True (one step per episode)
    Records last call kwargs and call count.
    """
    def __init__(self, env, rewards=None):
        self.env = env
        self.rewards = rewards or [1.0]
        self.last_kwargs = None
        self.calls = 0

    def collect_trajectory(self, agent, num_trajectories=1, min_num_steps=None):
        self.calls += 1
        self.last_kwargs = {
            "agent": agent,
            "num_trajectories": num_trajectories,
            "min_num_steps": min_num_steps,
        }

        # Treat each "trajectory" as a single step that ends immediately.
        k = min(num_trajectories, len(self.rewards))
        rew = torch.tensor(self.rewards[:k], dtype=torch.float32).view(-1, 1)  # (B,1)
        done = torch.ones(k, 1, dtype=torch.bool)                              # (B,1)

        return {"reward": rew, "done": done}

@pytest.fixture
def env():
    return FakeEnv(num_envs=2)


@pytest.fixture
def logger():
    return FakeLogger()


@pytest.fixture
def agent():
    # Nested structure so deepcopy is meaningful
    return {"weights": {"layer1": [1, 2, 3]}}


@pytest.fixture
def make_collector(monkeypatch):
    """
    Monkeypatch MODULE.ParallelCollector to return a FakeCollector with chosen rewards.
    Usage:
        make_collector(rewards=[...])  # patches the constructor
        reval = MODULE.RewardEvaluator(... )  # creates the FakeCollector
        collector = reval.collector  # access the instance
    """
    def _factory(rewards):
        def _ctor(env):
            return FakeCollector(env, rewards=rewards)
        monkeypatch.setattr(MODULE, "ParallelCollector", _ctor)
    return _factory


@pytest.fixture
def rewards_simple():
    return [10.0, 20.0, 30.0]

# ---------------------------
# Tests for Evaluator
# ---------------------------
def test_no_eval_on_first_iter_when_freq_gt1():
    ev = MODULE.Evaluator(eval_freq=5)
    assert ev.last_evaluation_iteration == 0
    assert ev._should_evaluate(0) is False
    # state unchanged
    assert ev.last_evaluation_iteration == 0


@pytest.mark.parametrize("freq, total_iters, expected_true_idxs", [
    # For freq=1: evaluate every step (0-based indices)
    (1, 6, [0, 1, 2, 3, 4, 5]),
    # For freq=3 across 0..8: true at 2,5,8
    (3, 9, [2, 5, 8]),
    # For freq=5 across 0..19: true at 4,9,14,19
    (5, 20, [4, 9, 14, 19]),
])
def test_should_evaluate_schedule(freq, total_iters, expected_true_idxs):
    ev = MODULE.Evaluator(eval_freq=freq)
    actual_true_idxs = []
    for i in range(total_iters):
        if ev._should_evaluate(i):
            actual_true_idxs.append(i)
    assert actual_true_idxs == expected_true_idxs


def test_no_false_positives_between_multiples():
    ev = MODULE.Evaluator(eval_freq=4)
    # 0..7 -> only 3 and 7 should be True
    results = [ev._should_evaluate(i) for i in range(8)]
    assert results == [False, False, False, True, False, False, False, True]


def test_last_evaluation_iteration_updates_on_true_only():
    ev = MODULE.Evaluator(eval_freq=5)
    # i=0..3 -> no eval; last stays 0
    for i in range(4):
        assert ev._should_evaluate(i) is False
        assert ev.last_evaluation_iteration == 0

    # i=4 -> eval; last becomes (i+1)=5
    assert ev._should_evaluate(4) is True
    assert ev.last_evaluation_iteration == 5

    # i=5..8 -> no eval; last remains 5
    for i in range(5, 9):
        assert ev._should_evaluate(i) is False
        assert ev.last_evaluation_iteration == 5

    # i=9 -> eval; last becomes 10
    assert ev._should_evaluate(9) is True
    assert ev.last_evaluation_iteration == 10


def test_close_is_noop():
    ev = MODULE.Evaluator(eval_freq=3)
    # Just ensure it doesn't raise
    ev.close()


def test_evaluate_is_callable_and_noop():
    ev = MODULE.Evaluator()
    # Should not raise even though it does nothing
    ev.evaluate(agent=None, iteration=0)

# ---------------------------
# Tests for RewardEvaluator
# ---------------------------
def test_initialization_sets_fields(env, logger, make_collector):
    make_collector([1.0])
    reval = MODULE.RewardEvaluator(
        env=env, num_episodes=3, logger=logger, keep_best=True, eval_freq=5, deterministic=True
    )
    assert reval.env is env
    assert reval.num_env == env.num_envs
    assert reval.num_episodes == 3
    assert reval.logger is logger
    assert reval.keep_best is True
    assert reval.eval_freq == 5
    assert reval.deterministic is True
    assert reval.best_reward == float("-inf")
    assert reval.best_agent is None
    assert hasattr(reval, "collector")


def test_evaluation_respects_schedule(env, logger, agent, make_collector, rewards_simple):
    make_collector(rewards_simple)
    reval = MODULE.RewardEvaluator(env=env, num_episodes=3, logger=logger, keep_best=False, eval_freq=3)

    # iterations: 0..4 ; only i=2 should trigger (0-based, due to +1)
    for i in range(5):
        reval.evaluate(agent, iteration=i, is_last=False)

    collector = reval.collector
    # One call to collector, at i=2
    assert collector.calls == 1
    assert collector.last_kwargs["num_trajectories"] == 3

    # Logger got 4 scalar logs (avg, std, max, min) once
    names = [t[0] for t in logger.scalars]
    assert names.count("evaluation_reward") == 1
    assert names.count("evaluation_reward_std") == 1
    assert names.count("evaluation_reward_max") == 1
    assert names.count("evaluation_reward_min") == 1


def test_is_last_forces_evaluation(env, logger, agent, make_collector, rewards_simple):
    make_collector(rewards_simple)
    reval = MODULE.RewardEvaluator(env=env, num_episodes=2, logger=logger, keep_best=False, eval_freq=1000)

    collector = reval.collector
    # Not on schedule, but is_last=True should still evaluate
    reval.evaluate(agent, iteration=7, is_last=True)
    assert collector.calls == 1


def test_logging_values_are_correct(env, logger, agent, make_collector):
    rewards = [5.0, 7.0, 9.0, 9.0]
    make_collector(rewards)
    reval = MODULE.RewardEvaluator(env=env, num_episodes=len(rewards), logger=logger, keep_best=False, eval_freq=1)

    reval.evaluate(agent, iteration=10)

    # Collect values
    vals = {name: value for (name, value, _) in logger.scalars}
    exp_avg = float(np.mean(rewards))
    exp_std = float(np.std(rewards))
    exp_max = float(np.max(rewards))
    exp_min = float(np.min(rewards))

    assert math.isclose(vals["evaluation_reward"], exp_avg, rel_tol=1e-6)
    assert math.isclose(vals["evaluation_reward_std"], exp_std, rel_tol=1e-6)
    assert math.isclose(vals["evaluation_reward_max"], exp_max, rel_tol=1e-6)
    assert math.isclose(vals["evaluation_reward_min"], exp_min, rel_tol=1e-6)

    # Iteration forwarded
    iters = {it for (_, _, it) in logger.scalars}
    assert 10 in iters


def test_keep_best_agent_updates_and_deepcopies(env, logger, agent, make_collector):
    # First eval: rewards avg = 1.0 -> becomes best
    make_collector([1.0, 1.0])
    reval = MODULE.RewardEvaluator(env=env, num_episodes=2, logger=logger, keep_best=True, eval_freq=1)

    reval.evaluate(agent, iteration=0)
    assert reval.best_reward == 1.0
    assert reval.best_agent is not None
    # Ensure deepcopy was used
    assert reval.best_agent is not agent
    # Mutate original; best_agent should be unchanged
    agent["weights"]["layer1"][0] = 999
    assert reval.best_agent["weights"]["layer1"][0] != 999

    # Second eval: better avg -> updates best
    # Patch collector to new rewards
    make_collector([2.0, 2.0])
    reval.collector = MODULE.ParallelCollector(env)
    reval.evaluate(agent, iteration=1)
    assert reval.best_reward == 2.0


def test_keep_best_agent_allows_equal_or_better(env, logger, agent, make_collector):
    # Start with avg 5.0
    make_collector([5.0, 5.0])
    reval = MODULE.RewardEvaluator(env=env, num_episodes=2, logger=logger, keep_best=True, eval_freq=1)
    reval.evaluate(agent, iteration=0)
    first_best = reval.best_agent

    # Equal performance should also update (>=)
    make_collector([5.0, 5.0])
    reval.collector = MODULE.ParallelCollector(env)
    reval.evaluate(agent, iteration=1)
    # best_agent should be replaced with a new deepcopy
    assert reval.best_agent is not first_best


@pytest.mark.parametrize("keep_best, has_logger, set_best", [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
])
def test_close_saves_agent_only_when_all_conditions_met(
    env, logger, agent, make_collector, keep_best, has_logger, set_best
):
    make_collector([3.0, 3.0])
    reval = MODULE.RewardEvaluator(
        env=env,
        num_episodes=2,
        logger=logger if has_logger else None,
        keep_best=keep_best,
        eval_freq=1,
    )

    if set_best:
        # Force a best agent to exist
        reval.evaluate(agent, iteration=0)

    reval.close()

    should_save = keep_best and has_logger and set_best
    assert (len(logger.saved) == 1) == should_save
    if should_save:
        saved_agent, path = logger.saved[0]
        assert path == "agent-best.pt"
        assert saved_agent is not None


def test_num_episodes_is_forwarded_to_collector(env, logger, agent, make_collector):
    make_collector([1.0, 2.0, 3.0, 4.0])
    reval = MODULE.RewardEvaluator(env=env, num_episodes=4, logger=logger, keep_best=False, eval_freq=1)
    reval.evaluate(agent, iteration=0)

    collector = reval.collector
    assert collector.last_kwargs["num_trajectories"] == 4

# ---------------------------
# Tests for NumberOfSteps
# ---------------------------
def test_initialization_sets_fields(env, logger, make_collector):
    make_collector([1.0])
    neval = MODULE.NumberOfStepsEvaluator(
        env=env,
        reward_threshold=10.0,
        num_episodes=3,  # note the typo in ctor arg name
        logger=logger,
        keep_best=True,
        eval_freq=5,
        deterministic=True,
    )
    assert neval.env is env
    assert neval.reward_threshold == 10.0
    assert neval.num_episodes == 3   # attribute spelled correctly in class
    assert neval.logger is logger
    assert neval.keep_best is True
    assert neval.eval_freq == 5
    assert neval.deterministic is True
    assert hasattr(neval, "collector")
    # best_timestep should start "unset"
    # (the current code uses int("inf") which raises; see logic notes below)
    # This assertion expects a sane initialization:
    assert math.isinf(neval.best_timestep)


def test_evaluation_respects_schedule(env, logger, agent, make_collector):
    # Rewards below threshold -> no best_timestep update
    make_collector([1.0, 2.0, 3.0])
    neval = MODULE.NumberOfStepsEvaluator(
        env=env, reward_threshold=10.0, num_episodes=3, logger=logger, keep_best=False, eval_freq=3
    )

    # iterations 0..4; only i=2 should evaluate (0-based with +1 inside _should_evaluate)
    for i in range(5):
        neval.evaluate(agent, iteration=i, is_last=False)

    collector = neval.collector
    assert collector.calls == 1
    assert collector.last_kwargs["num_trajectories"] == 3

    # Logger logs at the time of evaluation; value should still be inf because threshold not met
    vals = {name: v for (name, v, _) in logger.scalars}
    assert "evaluation_numsteps" in vals
    assert math.isinf(vals["evaluation_numsteps"])


def test_is_last_forces_evaluation(env, logger, agent, make_collector):
    make_collector([100.0])  # above threshold
    neval = MODULE.NumberOfStepsEvaluator(
        env=env, reward_threshold=10.0, num_episodes=1, logger=logger, keep_best=False, eval_freq=1000
    )

    # Not on schedule, but is_last=True should still evaluate
    neval.evaluate(agent, iteration=7, is_last=True)
    assert neval.collector.calls == 1
    # Should update best_timestep to 7 (assuming 0-based iteration semantics)
    assert neval.best_timestep == 7


def test_threshold_not_met_does_not_update_best(env, logger, agent, make_collector):
    make_collector([4.0, 5.0])  # avg = 4.5 < 10
    neval = MODULE.NumberOfStepsEvaluator(
        env=env, reward_threshold=10.0, num_episodes=2, logger=logger, keep_best=True, eval_freq=1
    )

    neval.evaluate(agent, iteration=3)
    assert math.isinf(neval.best_timestep)
    assert neval.best_agent is None


def test_threshold_met_updates_once_and_uses_min_steps(env, logger, agent, make_collector):
    # First time threshold met at i=5
    make_collector([15.0, 15.0, 15.0])
    neval = MODULE.NumberOfStepsEvaluator(
        env=env, reward_threshold=10.0, num_episodes=3, logger=logger, keep_best=True, eval_freq=1
    )
    neval.evaluate(agent, iteration=5)
    first_best = neval.best_timestep
    assert first_best == 5
    assert neval.best_agent is not None  # deepcopy expected

    # Later evaluations also meet threshold but occur at *more* steps
    # best_timestep should remain the *minimum* number of steps
    make_collector([20.0, 20.0, 20.0])
    neval.collector = MODULE.ParallelCollector(neval.env)
    neval.evaluate(agent, iteration=8)
    assert neval.best_timestep == 5  # must *not* overwrite with worse (larger) step count

    # If a later evaluation somehow has a *smaller* iteration (e.g., different run/reset),
    # it should be allowed to improve the best_timestep.
    # We simulate by directly passing a smaller iteration.
    neval.evaluate(agent, iteration=4, is_last=True)
    assert neval.best_timestep == 4


def test_deepcopy_when_keep_best(env, logger, agent, make_collector):
    make_collector([11.0, 12.0])
    neval = MODULE.NumberOfStepsEvaluator(
        env=env, reward_threshold=10.0, num_episodes=2, logger=logger, keep_best=True, eval_freq=1
    )
    neval.evaluate(agent, iteration=2)
    # Ensure deepcopy was used (mutating source shouldn't change saved agent)
    assert neval.best_agent is not agent
    agent["weights"]["layer1"][0] = 999
    assert neval.best_agent["weights"]["layer1"][0] != 999


def test_logging_value(env, logger, agent, make_collector):
    # First eval below threshold, second above threshold
    make_collector([0.0, 0.0])
    neval = MODULE.NumberOfStepsEvaluator(
        env=env, reward_threshold=5.0, num_episodes=2, logger=logger, keep_best=False, eval_freq=1
    )
    neval.evaluate(agent, iteration=0)
    make_collector([10.0, 10.0])
    neval.collector = MODULE.ParallelCollector(neval.env)
    neval.evaluate(agent, iteration=3)

    # Grab last logged value; should be 3
    last = logger.scalars[-1]
    name, value, it = last
    assert name == "evaluation_numsteps"
    assert value == 3
    assert it == 3


@pytest.mark.parametrize("keep_best, has_logger, set_best", [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
])
def test_close_saves_agent_only_when_all_conditions_met(
    env, logger, agent, make_collector, keep_best, has_logger, set_best
):
    make_collector([10.0, 10.0])
    neval = MODULE.NumberOfStepsEvaluator(
        env=env,
        reward_threshold=5.0,
        num_episodes=2,
        logger=logger if has_logger else None,
        keep_best=keep_best,
        eval_freq=1,
    )

    if set_best:
        neval.evaluate(agent, iteration=1)

    neval.close()

    should_save = keep_best and has_logger and set_best
    assert (len(logger.saved) == 1) == should_save
    if should_save:
        saved_agent, path = logger.saved[0]
        assert path == "agent-best.pt"
        assert saved_agent is not None