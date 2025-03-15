import torch
from prt_rl.common.qtable import QTable

def test_qtable_initialization():
    qtable = QTable(
        state_dim=3,
        action_dim=2,
    )

    assert qtable.q_table.shape == (1, 3, 2)

def test_qtable_initial_value():
    qtable = QTable(
        state_dim=3,
        action_dim=2,
        initial_value=1.0,
    )

    assert torch.allclose(qtable.q_table, torch.ones((1, 3, 2)))

def test_qtable_update():
    qtable = QTable(
        state_dim=3,
        action_dim=2,
    )

    state = torch.tensor([[1]], dtype=torch.int)
    action = torch.tensor([[0]], dtype=torch.int)
    qval = torch.tensor([[0.3]], dtype=torch.float)
    assert state.shape == torch.Size([1, 1])
    assert action.shape == torch.Size([1, 1])
    assert qval.shape == torch.Size([1, 1])

    qtable.update_q_value(state=state, action=action, q_value=qval)

    assert qtable.q_table.shape == (1, 3, 2)
    assert qtable.q_table[0, 1, 0] == 0.3

def test_qtable_gets():
    qtable = QTable(
        state_dim=3,
        action_dim=2,
    )

    state = torch.tensor([[1]], dtype=torch.int)
    action = torch.tensor([[0]], dtype=torch.int)
    qval = torch.tensor([[0.3]], dtype=torch.float)
    qtable.update_q_value(state=state, action=action, q_value=qval)
    qtable.update_q_value(state=state, action=torch.tensor([[1]], dtype=torch.int), q_value=torch.tensor([[0.2]]))

    action_vals = qtable.get_action_values(state=state)
    assert action_vals.shape == (1, 2)
    assert torch.allclose(action_vals, torch.tensor([[0.3, 0.2]]))

    state_action_val = qtable.get_state_action_value(state=state, action=torch.tensor([[0]], dtype=torch.int))
    assert state_action_val.shape == (1, 1)
    assert state_action_val[0] == 0.3

def test_qtable_track_visits():
    qtable = QTable(
        state_dim=3,
        action_dim=2,
        track_visits=True,
    )

    state = torch.tensor([[1]], dtype=torch.int)
    action = torch.tensor([[0]], dtype=torch.int)

    qtable.update_visits(state=state, action=action)

    assert qtable.visit_table.shape == torch.Size([1, 3, 2])
    assert qtable.visit_table[0, 1, 0] == 1
    assert qtable.get_visit_count(state, action)[0] == 1
