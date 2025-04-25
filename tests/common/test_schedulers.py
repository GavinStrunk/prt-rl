import pytest
import prt_rl.common.schedulers as sch

class EpsilonGreedy:
    def __init__(self):
        self.epsilon = 1.0


def test_linear_schedule():
    # Schedules parameter down
    eg = EpsilonGreedy()
    s = sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.2, end_value=0.1, num_steps=10)

    s.update(current_step=0)
    assert eg.epsilon == 0.2

    s.update(current_step=5)
    assert eg.epsilon == pytest.approx(0.15)

    s.update(current_step=10)
    assert eg.epsilon == 0.1

    s.update(current_step=15)
    assert eg.epsilon == 0.1

    # Schedules parameter up
    s = sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.0, end_value=1.0, num_steps=10)
    s.update(current_step=0)
    assert eg.epsilon == 0.0
    s.update(current_step=5)
    assert eg.epsilon == pytest.approx(0.5)
    s.update(current_step=10)
    assert eg.epsilon == 1.0


def test_linear_invalid_inputs():
    eg = EpsilonGreedy()
    # Number of episodes must be greater than 0
    with pytest.raises(AssertionError):
        sch.LinearScheduler(obj=eg, parameter_name='epsilon', start_value=0.1, end_value=0.3, num_steps=0)