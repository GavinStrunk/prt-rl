import torch
from torch.distributions import Normal
from prt_rl.model_based.planners.cross_entropy import CrossEntropyMethodPlanner

def test_cold_start_mean_and_std():
    action_mins = torch.tensor([[-1.0], [-1.0]])
    action_maxs = torch.tensor([[1.0], [2.0]])

    planner = CrossEntropyMethodPlanner(
        action_mins=action_mins,
        action_maxs=action_maxs,
    )

    planner._initialize_distribution()

    assert isinstance(planner.distribution, Normal)
    assert planner.distribution.loc.shape == (planner.planning_horizon, action_mins.shape[0])
    assert planner.distribution.scale.shape == (planner.planning_horizon, action_mins.shape[0])
    torch.allclose(planner.distribution.mean, torch.tensor([[0.0, 0.5]]).expand(planner.planning_horizon, -1))

def test_distribution_sampling_shape():
    action_mins = torch.tensor([[-1.0], [-1.0]])
    action_maxs = torch.tensor([[1.0], [2.0]])
    H = 5
    num_action_sequences = 10

    planner = CrossEntropyMethodPlanner(
        action_mins=action_mins,
        action_maxs=action_maxs,
        num_action_sequences=num_action_sequences,
        planning_horizon=H,
    )

    planner._initialize_distribution()

    samples = planner.distribution.rsample((num_action_sequences,))  # (B, H, A)
    assert samples.shape == (num_action_sequences, H, action_mins.shape[0])

def test_planning():
    action_mins = torch.tensor([[-1.0], [-1.0]])
    action_maxs = torch.tensor([[1.0], [2.0]])
    H = 5
    num_action_sequences = 20
    num_elites = 5
    num_iterations = 3

    planner = CrossEntropyMethodPlanner(
        action_mins=action_mins,
        action_maxs=action_maxs,
        num_action_sequences=num_action_sequences,
        num_elites=num_elites,
        num_iterations=num_iterations,
        planning_horizon=H,
    )

    state = torch.zeros((1, 3))  # Dummy state with batch size 1 and state dimension 3

    best_action = planner.plan(None, None, state)

    assert best_action.shape == (1, action_mins.shape[0])