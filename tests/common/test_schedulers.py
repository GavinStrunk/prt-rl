import pytest
import prt_rl.common.schedulers as sch

def test_linear_schedule():
    # Schedules parameter down
    s = sch.LinearScheduler('a', start_value=0.2, end_value=0.1, num_episodes=10)
    assert s.update(iteration_number=0) == 0.2
    assert s.update(iteration_number=5) == pytest.approx(0.15)
    assert s.update(iteration_number=10) == 0.1
    assert s.update(iteration_number=15) == 0.1

    # Schedules parameter up
    s = sch.LinearScheduler('a', start_value=0.0, end_value=1.0, num_episodes=10)
    assert s.update(iteration_number=0) == 0.0
    assert s.update(iteration_number=10) == 1.0
    assert s.update(iteration_number=5) == 0.5


def test_linear_invalid_inputs():
    # Number of episodes must be greater than 0
    with pytest.raises(AssertionError):
        sch.LinearScheduler('a', start_value=0.1, end_value=0.3, num_episodes=0)