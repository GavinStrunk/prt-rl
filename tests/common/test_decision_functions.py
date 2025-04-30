import torch
import pytest
import prt_rl.common.decision_functions as df

def check_input_dimensions(test_func):
    # No batch dimension
    action_pmf = torch.tensor([0.5, 0.5])
    assert action_pmf.shape == (2,)
    with pytest.raises(ValueError):
        test_func(action_pmf)

    # Too many action dimensions
    action_pmf = torch.tensor([[[0.5], [0.5]]])
    assert action_pmf.shape == (1, 2, 1)
    with pytest.raises(ValueError):
        test_func(action_pmf)


def test_stochastic_selection():
    # Make fake action pmf with batch size 1
    action_pmf = torch.tensor([[0.5, 0.5]])
    assert action_pmf.shape == (1, 2)

    torch.manual_seed(0)
    actions = df.stochastic_selection(action_pmf)
    assert actions.shape == (1, 1)
    assert actions[0] == 1

    # Sample with multiple environments
    action_pmf = torch.tensor([[0.5, 0.5], [0.7, 0.3], [1.0, 0.0]])
    assert action_pmf.shape == (3, 2)
    actions = df.stochastic_selection(action_pmf)
    assert actions.shape == (3, 1)
    assert actions[0] == 1
    assert actions[1] == 1
    assert actions[2] == 0

def test_stochastic_selection_invalid_inputs():
    check_input_dimensions(df.stochastic_selection)

    # Not a valid pmf: does not sum to 1.0
    action_pmf = torch.tensor([[0.5, 0.4]])
    with pytest.raises(ValueError):
        df.stochastic_selection(action_pmf)

    # Negative probabilities are not valid
    action_pmf = torch.tensor([[0.5, 0.6, -0.1]])
    with pytest.raises(ValueError):
        df.stochastic_selection(action_pmf)

def test_greedy_decision_function():
    action_vals = torch.tensor([[0.1, 0.2, 0.15]])
    assert action_vals.shape == torch.Size([1, 3])
    dfunc = df.Greedy()

    torch.manual_seed(0)
    action = dfunc.select_action(action_vals)
    assert action.shape == torch.Size([1, 1])
    assert action[0] == 1

    # Multiple max values a random one is chosen
    action_vals = torch.tensor([[0.2, 0.2, 0.2]])
    action = dfunc.select_action(action_vals)
    assert action.shape == torch.Size([1, 1])
    assert action[0] == 2

    # Multiple environments
    action_vals = torch.tensor([[0.1, 0.2, 0.15], [0.1, 0.2, 0.15]])
    action = dfunc.select_action(action_vals)
    assert action.shape == torch.Size([2, 1])
    assert action[0] == 1
    assert action[1] == 1

def test_greedy_decision_function_invalid_inputs():
    dfunc = df.Greedy()
    check_input_dimensions(dfunc.select_action)

def test_epsilon_greedy_decision_function():
    action_vals = torch.tensor([[0.1, 0.2, 0.15]])

    torch.manual_seed(0)
    # Check it is greedy when epsilon is 0.0
    dfunc = df.EpsilonGreedy(epsilon=0.0)
    action = dfunc.select_action(action_vals)
    assert action.shape == torch.Size([1, 1])
    assert action[0] == 1

    # Check updating epsilon
    dfunc.set_parameter(name="epsilon", value=0.5)
    assert dfunc.epsilon == 0.5
    action = dfunc.select_action(action_vals)
    assert action[0] == 1

    # Check multiple environments
    torch.manual_seed(1)
    action_vals = torch.tensor([[0.1, 0.2, 0.15], [0.1, 0.2, 0.15]])
    action = dfunc.select_action(action_vals)
    assert action.shape == torch.Size([2, 1])
    assert action[0] == 1
    assert action[1] == 0

def test_softmax_decision_function():
    action_vals = torch.tensor([[0.1, 0.2, 0.15]])
    dfunc = df.Softmax(tau=1.0)

    torch.manual_seed(2)
    action = dfunc.select_action(action_vals)
    assert action == 0

    assert dfunc.tau == 1.0
    dfunc.set_parameter(name="tau", value=0.1)
    assert dfunc.tau == 0.1
