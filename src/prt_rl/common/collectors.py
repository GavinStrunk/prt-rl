import torch
from typing import Dict

class Collector:
    def __init__(self, 
                 env, 
                 ):
        self.env = env

        # Detect the number of environments
        state, _ = self.env.reset()
        self.num_envs = state.shape[0]

    def collect(self, 
                policy, 
                num_steps: int
                ) -> Dict[str, torch.Tensor]:
        
        steps_per_env = num_steps // self.num_envs

        # Reset all environments
        state, _ = self.env.reset()

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        for _ in range(steps_per_env):
            with torch.no_grad():
                action = policy(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = torch.logical_or(terminated, truncated)

            # Append data
            states.append(state)
            actions.append(action)
            next_states.append(torch.from_numpy(next_state).float())
            rewards.append(torch.from_numpy(reward).float())
            dones.append(torch.from_numpy(done).bool())

            # Reset only where done
            if torch.any(done):
                next_state[done] = self.env.reset(seed=None)[0][done]


        return {
            "state": torch.cat(states),
            "action": torch.cat(actions),
            "next_state": torch.cat(next_states),
            "reward": torch.cat(rewards),
            "done": torch.cat(dones),
        }