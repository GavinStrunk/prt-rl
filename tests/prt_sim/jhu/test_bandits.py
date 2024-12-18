import numpy as np
import pytest
from prt_sim.jhu.bandits import KArmBandits

def test_bandits_are_positive():
    with pytest.raises(AssertionError):
        KArmBandits(num_bandits=-1)

    with pytest.raises(AssertionError):
        KArmBandits(num_bandits=0)

def test_bandits_construction():
    env = KArmBandits()
    assert env.bandit_probs.shape == (10,)

    env = KArmBandits(num_bandits=2)
    assert env.bandit_probs.shape == (2,)

def test_bandits_get_states():
    # The bandits environment does not have states
    env = KArmBandits()
    states = env.get_number_of_states()
    assert states == 0

def test_bandits_get_actions():
    env = KArmBandits()
    actions = env.get_number_of_actions()
    assert actions  == 10

def test_bandits_implicit_reset():
    env = KArmBandits(num_bandits=3)
    np.random.seed(0)
    env.reset()

    np.testing.assert_allclose(env.bandit_probs, np.array([1.764, 0.400, 0.979]), atol=1e-3)

def test_bandits_get_optimal_bandit():
    env = KArmBandits(num_bandits=3)
    env.reset(seed=2)

    opt_band = env.get_optimal_bandit()
    assert opt_band  == 1

    env.reset(seed=0)
    opt_band = env.get_optimal_bandit()
    assert opt_band == 0

def test_bandits_bad_action():
    env = KArmBandits(num_bandits=3)
    env.reset()

    with pytest.raises(AssertionError):
        env.execute_action(action=-1)

    with pytest.raises(AssertionError):
        env.execute_action(action=3)

def test_bandits_execute_action():
    env = KArmBandits(num_bandits=3)
    np.random.seed(0)
    env.reset()

    _, reward, _ = env.execute_action(action=1)
    assert pytest.approx(reward, abs=1e-2) == 2.64