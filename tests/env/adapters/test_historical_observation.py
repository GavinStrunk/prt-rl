import torch

from prt_rl.env.adapters import HistoricalObservationAdapter
from prt_rl.env.interface import EnvParams, EnvironmentInterface


class DummyEnv(EnvironmentInterface):
    def __init__(self):
        super().__init__(num_envs=1)
        self._params = EnvParams(
            action_len=2,
            action_continuous=True,
            action_min=[-2.0, -3.0],
            action_max=[2.0, 3.0],
            observation_shape=(2,),
            observation_continuous=True,
            observation_min=[-1.0, -1.0],
            observation_max=[1.0, 1.0],
        )
        self._obs = [
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        ]
        self._idx = 0

    def get_parameters(self):
        return self._params

    def reset(self, seed=None):
        self._idx = 0
        return self._obs[self._idx], {}

    def step(self, action):
        self._idx = min(self._idx + 1, len(self._obs) - 1)
        obs = self._obs[self._idx]
        reward = torch.zeros((1, 1), dtype=torch.float32)
        done = torch.zeros((1, 1), dtype=torch.bool)
        return obs, reward, done, {}


def test_historical_observation_adapter_obs_only():
    env = HistoricalObservationAdapter(DummyEnv(), num_steps=3, include_actions=False)
    params = env.get_parameters()

    assert params.observation_shape == (6,)
    assert params.observation_min == [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    assert params.observation_max == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    obs, _ = env.reset()
    assert torch.allclose(obs, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0]]))

    obs, _, _, _ = env.step(torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(obs, torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]]))


def test_historical_observation_adapter_obs_and_actions():
    env = HistoricalObservationAdapter(DummyEnv(), num_steps=3, include_actions=True)
    params = env.get_parameters()

    assert params.observation_shape == (10,)
    assert params.observation_min == [-1.0, -1.0, -2.0, -3.0, -1.0, -1.0, -2.0, -3.0, -1.0, -1.0]
    assert params.observation_max == [1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0]

    obs, _ = env.reset()
    assert torch.allclose(obs, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]]))

    obs, _, _, _ = env.step(torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(obs, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 10.0, 20.0, 3.0, 4.0]]))


def test_historical_observation_adapter_obs_and_actions_with_appended_last_action():
    env = HistoricalObservationAdapter(
        DummyEnv(),
        num_steps=3,
        include_actions=True,
        append_last_action=True,
    )
    params = env.get_parameters()

    assert params.observation_shape == (12,)
    assert params.observation_min == [-1.0, -1.0, -2.0, -3.0, -1.0, -1.0, -2.0, -3.0, -1.0, -1.0, -2.0, -3.0]
    assert params.observation_max == [1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0]

    obs, _ = env.reset()
    assert torch.allclose(obs, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]]))

    obs, _, _, _ = env.step(torch.tensor([[10.0, 20.0]]))
    # [o_{t-2}, a_{t-2}, o_{t-1}, a_{t-1}, o_t, a_{t-1}]
    assert torch.allclose(obs, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 10.0, 20.0]]))
