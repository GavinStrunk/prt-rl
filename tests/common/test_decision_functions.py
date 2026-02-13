import pytest
import torch

import prt_rl.common.decision_functions as df


def check_invalid_dimensions(test_func):
    # Too many action dimensions
    action_values = torch.tensor([[[0.5], [0.5]]])
    assert action_values.shape == (1, 2, 1)
    with pytest.raises(ValueError):
        test_func(action_values)


def test_stochastic_selection():
    action_pmf = torch.tensor([[0.5, 0.5]])
    torch.manual_seed(0)
    actions = df.stochastic_selection(action_pmf)
    assert actions.shape == (1, 1)
    assert actions[0].item() in [0, 1]

    action_pmf = torch.tensor([[0.5, 0.5], [0.7, 0.3], [1.0, 0.0]])
    torch.manual_seed(2)
    actions = df.stochastic_selection(action_pmf)
    assert actions.shape == (3, 1)
    assert actions[2].item() == 0


def test_stochastic_selection_invalid_inputs():
    # stochastic_selection still requires batched 2D input
    with pytest.raises(ValueError):
        df.stochastic_selection(torch.tensor([0.5, 0.5]))

    check_invalid_dimensions(df.stochastic_selection)

    with pytest.raises(ValueError):
        df.stochastic_selection(torch.tensor([[0.5, 0.4]]))

    with pytest.raises(ValueError):
        df.stochastic_selection(torch.tensor([[0.5, 0.6, -0.1]]))


def test_greedy_decision_function_batched_and_unbatched():
    dfunc = df.Greedy()

    action_vals = torch.tensor([[0.1, 0.2, 0.15], [0.1, 0.3, 0.2]])
    action = dfunc.select_action(action_vals)
    assert action.shape == (2, 1)
    assert action[0].item() == 1
    assert action[1].item() == 1

    action_vals_1d = torch.tensor([0.1, 0.2, 0.15])
    action_1d = dfunc.select_action(action_vals_1d)
    assert action_1d.shape == (1,)
    assert action_1d.item() == 1


def test_greedy_decision_function_invalid_inputs():
    dfunc = df.Greedy()
    check_invalid_dimensions(dfunc.select_action)


def test_epsilon_greedy_decision_function_batched_and_unbatched():
    dfunc = df.EpsilonGreedy(epsilon=0.0)

    batched_vals = torch.tensor([[0.1, 0.2, 0.15], [0.3, 0.1, 0.2]])
    action = dfunc.select_action(batched_vals)
    assert action.shape == (2, 1)
    assert action[0].item() == 1
    assert action[1].item() == 0

    unbatched_vals = torch.tensor([0.1, 0.2, 0.15])
    action_1d = dfunc.select_action(unbatched_vals)
    assert action_1d.shape == (1,)
    assert action_1d.item() == 1

    dfunc.set_parameter(name="epsilon", value=0.5)
    assert dfunc.epsilon == 0.5


def test_softmax_decision_function_batched_and_unbatched():
    dfunc = df.Softmax(tau=1.0)

    batched_vals = torch.tensor([[0.1, 0.2, 0.15], [0.9, 0.1, 0.2]])
    torch.manual_seed(0)
    action = dfunc.select_action(batched_vals)
    assert action.shape == (2, 1)
    assert 0 <= action[0].item() < 3
    assert 0 <= action[1].item() < 3

    unbatched_vals = torch.tensor([0.1, 0.2, 0.15])
    torch.manual_seed(0)
    action_1d = dfunc.select_action(unbatched_vals)
    assert action_1d.shape == (1,)
    assert 0 <= action_1d.item() < 3

    dfunc.set_parameter(name="tau", value=0.1)
    assert dfunc.tau == 0.1


def test_softmax_invalid_tau():
    dfunc = df.Softmax(tau=0.0)
    with pytest.raises(ValueError):
        dfunc.select_action(torch.tensor([0.1, 0.2]))


def test_ucb_decision_function_batched_and_unbatched():
    dfunc = df.UpperConfidenceBound(c=1.0, t=10.0)

    batched_vals = torch.tensor([[0.1, 0.2, 0.15], [0.9, 0.1, 0.2]])
    batched_counts = torch.tensor([[1.0, 2.0, 1.0], [5.0, 1.0, 3.0]])
    action = dfunc.select_action(batched_vals, batched_counts)
    assert action.shape == (2, 1)
    assert 0 <= action[0].item() < 3
    assert 0 <= action[1].item() < 3

    unbatched_vals = torch.tensor([0.1, 0.2, 0.15])
    unbatched_counts = torch.tensor([1.0, 2.0, 1.0])
    action_1d = dfunc.select_action(unbatched_vals, unbatched_counts)
    assert action_1d.shape == (1,)
    assert 0 <= action_1d.item() < 3

    # Without action counts, UCB falls back to greedy selection.
    fallback_action = dfunc.select_action(torch.tensor([0.1, 0.2, 0.15]))
    assert fallback_action.shape == (1,)
    assert fallback_action.item() == 1
