import torch
from prt_rl.utils.qtable import QTable

def test_qtable_initialization():
    qtable = QTable(
        state_dim=3,
        action_dim=2,
    )

    assert qtable.q_table.shape == (3, 2)

def test_qtable_initial_value():
    qtable = QTable(
        state_dim=3,
        action_dim=2,
        initial_value=1.0,
    )

    assert qtable.q_table.shape == (3, 2)
    assert torch.allclose(qtable.q_table, torch.ones((3, 2)))

def test_qtable_update():
    qtable = QTable(
        state_dim=3,
        action_dim=2,
    )

    qtable.update_q_value(state=1, action=0, q_value=0.3)

    assert qtable.q_table.shape == (3, 2)
    assert qtable.q_table[1, 0] == 0.3

def test_qtable_gets():
    qtable = QTable(
        state_dim=3,
        action_dim=2,
    )

    qtable.update_q_value(state=1, action=0, q_value=0.3)
    qtable.update_q_value(state=1, action=1, q_value=0.2)

    assert qtable.get_action_values(state=1).shape == (2,)
    assert torch.allclose(qtable.get_action_values(state=1), torch.tensor([0.3, 0.2]))

    assert qtable.get_state_action_value(state=1, action=0).shape == torch.Size([])
    assert qtable.get_state_action_value(state=1, action=1) == 0.2