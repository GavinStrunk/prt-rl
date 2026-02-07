from prt_rl.env.interface import EnvironmentInterface, EnvParams

class AdapterInterface(EnvironmentInterface):
    """
    Interface class for environment adapters that adapt an environment to a different interface.
    
    Args:
        env (EnvironmentInterface): The environment to adapt
    """
    def __init__(self, env: EnvironmentInterface):
        self.env = env
        self.env_params = self._adapt_params(env.get_parameters())

    def get_parameters(self):
        return self.env_params

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._adapt_obs(obs, info), self._adapt_info(None, obs, None, False, info)

    def step(self, action):
        raw_action = self._adapt_action(action)
        obs, reward, done, info = self.env.step(raw_action)
        adapted_info = self._adapt_info(action, obs, reward, done, info)
        return self._adapt_obs(obs, info), self._adapt_reward(reward, info), done, adapted_info

    # The following methods can be overridden by subclasses to adapt the parameters, observations, actions, rewards, and info dictionaries as needed. By default, they return the input unchanged.
    def _adapt_params(self, params: EnvParams): return params
    def _adapt_obs(self, obs, info): return obs
    def _adapt_action(self, action): return action
    def _adapt_reward(self, reward, info): return reward
    def _adapt_info(self, action, obs, reward, done, info): return info
