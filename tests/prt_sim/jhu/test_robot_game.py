import numpy as np
from prt_sim.jhu.robot_game import RobotGame

def test_reset_starting_state():
    env = RobotGame()
    state = env.reset()
    print(state)
    assert state == 8

def test_get_number_of_states():
    env = RobotGame()
    assert env.get_number_of_states() == 11

def test_get_number_of_actions():
    env = RobotGame()
    assert env.get_number_of_actions() == 4

def test_up_action():
    env = RobotGame()
    state = env.reset()
    state, reward, done = env.execute_action(RobotGame.UP)
    assert state == 5

def test_down_action():
    env = RobotGame()
    state = env.reset()
    state, reward, done = env.execute_action(RobotGame.UP)
    state, reward, done = env.execute_action(RobotGame.DOWN)
    assert state == 8

def test_trying_to_leave_bottom():
    env = RobotGame()
    state = env.reset()
    state, reward, done = env.execute_action(RobotGame.DOWN)
    assert state == 8

def test_trying_to_leave_top():
    env = RobotGame()
    state = env.reset()
    state, reward, done = env.execute_action(RobotGame.UP)
    state, reward, done = env.execute_action(RobotGame.UP)
    state, reward, done = env.execute_action(RobotGame.UP)
    assert state == 1

def test_trying_to_leave_left():
    env = RobotGame()
    state = env.reset()
    state, reward, done = env.execute_action(RobotGame.LEFT)
    assert state == 8

def test_trying_to_leave_right():
    env = RobotGame()
    state = env.reset()
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    assert state == 11

def test_trying_reach_empty_space():
    env = RobotGame()
    state = env.reset()

    # Teleport robot to state 6
    env.current_position = np.array([1, 1])
    assert env.get_state() == 6
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    assert state == 6

    state = env.reset()
    # Teleport robot to state 3
    env.current_position = np.array([2, 0])
    assert env.get_state() == 3
    state, reward, done = env.execute_action(RobotGame.DOWN)
    assert state == 3

    state = env.reset()
    # Teleport robot to state 10
    env.current_position = np.array([2, 2])
    assert env.get_state() == 10
    state, reward, done = env.execute_action(RobotGame.UP)
    assert state == 10

    state = env.reset()
    # Teleport robot to state 7
    env.current_position = np.array([3, 1])
    assert env.get_state() == 7
    state, reward, done = env.execute_action(RobotGame.LEFT)
    assert done == True
    assert state == 7

def test_reaches_goal():
    env = RobotGame()
    state = env.reset()
    state, reward, done = env.execute_action(RobotGame.UP)
    state, reward, done = env.execute_action(RobotGame.UP)
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    assert state == 4
    assert reward == 25
    assert done == True

def test_reaches_pit():
    env = RobotGame()
    state = env.reset()
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    state, reward, done = env.execute_action(RobotGame.RIGHT)
    state, reward, done = env.execute_action(RobotGame.UP)
    assert state == 7
    assert reward == -25
    assert done == True
