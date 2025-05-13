import pytest
import prt_rl.common.schedulers as sch

class EpsilonGreedy:
    def __init__(self):
        self.epsilon = 1.0


def test_linear_schedule():
    # Schedules parameter down
    eg = EpsilonGreedy()
    s = sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.2, end_value=0.1, interval=10)

    s.update(current_step=0)
    assert eg.epsilon == 0.2

    s.update(current_step=5)
    assert eg.epsilon == pytest.approx(0.15)

    s.update(current_step=10)
    assert eg.epsilon == 0.1

    s.update(current_step=15)
    assert eg.epsilon == 0.1

    # Schedules parameter up
    s = sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.0, end_value=1.0, interval=10)
    s.update(current_step=0)
    assert eg.epsilon == 0.0
    s.update(current_step=5)
    assert eg.epsilon == pytest.approx(0.5)
    s.update(current_step=10)
    assert eg.epsilon == 1.0


def test_linear_invalid_inputs():
    eg = EpsilonGreedy()
    # Number of episodes must be greater than 0
    with pytest.raises(ValueError):
        sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.1, end_value=0.3, interval=0)

def test_linear_schedule_interval():
    eg = EpsilonGreedy()
    s = sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.2, end_value=0.1, interval=(4,10))
    
    s.update(current_step=0)
    assert eg.epsilon == 0.2

    s.update(current_step=3)
    assert eg.epsilon == 0.2

    s.update(current_step=7)
    assert eg.epsilon == pytest.approx(0.15)

    s.update(current_step=10)
    assert eg.epsilon == 0.1
    s.update(current_step=15)
    assert eg.epsilon == 0.1

def test_definition_mismatch():
    eg = EpsilonGreedy()
    # Number of episodes must be greater than 0
    with pytest.raises(ValueError):
        sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.1, end_value=[0.3], interval=[(0, 10), (10, 20)])

def test_invalid_overlapping_intervals():
    eg = EpsilonGreedy()
    # Number of episodes must be greater than 0
    with pytest.raises(ValueError):
        sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.1, end_value=[0.3, 0.2], interval=[(0, 10), (5, 20)])

    s = sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.1, end_value=[0.3, 0.2], interval=[(0, 10), (10, 20)])
    assert s.rates == pytest.approx([0.02, -0.01])

def test_piecewise_linear_schedule():
    eg = EpsilonGreedy()
    s = sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=1.0, end_value=[0.4, 0.1], interval=[(4, 10), (15, 16)])

    assert s.rates == pytest.approx([-0.1, -0.3])

    s.update(current_step=0)
    assert eg.epsilon == 1.0
    s.update(current_step=4)
    assert eg.epsilon == 1.0
    s.update(current_step=7)
    assert eg.epsilon == pytest.approx(0.7)
    s.update(current_step=10)
    assert eg.epsilon == 0.4
    s.update(current_step=15)
    assert eg.epsilon == 0.4
    s.update(current_step=16)
    assert eg.epsilon == 0.1
    s.update(current_step=20)
    assert eg.epsilon == 0.1


def test_sequential_updates():
    eg = EpsilonGreedy()
    s = sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.2, end_value=0.1, interval=10)

    s.update(current_step=1)
    assert eg.epsilon == pytest.approx(0.19)
    for _ in range(10):
        s.update(current_step=1)
    assert eg.epsilon == pytest.approx(0.19)