import matplotlib.pyplot as plt
import numpy as np
from prt_rl.exact.sample_average import SampleAverage
from prt_rl.env.wrappers import JhuWrapper
from prt_sim.jhu.bandits import KArmBandits
import prt_sim.jhu.plot as pplt
from tqdm import tqdm

# Set the number of runs and episodes per run
num_runs = 100
num_episodes = 1000

# Initialize Environment
env = JhuWrapper(environment=KArmBandits())

actions = np.zeros((num_runs, num_episodes), dtype=int)
rewards = np.zeros((num_runs, num_episodes))
optimal_bandit = np.zeros(num_runs, dtype=int)
for run in tqdm(range(num_runs)):
    obs_td = env.reset()

    optimal_bandit[run] = obs_td['info', 'optimal_bandit'].item()
    agent = SampleAverage(env=env)
    for step in range(num_episodes):
        action_td = agent.policy.get_action(obs_td)

        actions[run][step] = action_td['action'].item()
        obs_td = env.step(action_td)
        agent.update_policy(obs_td)
        rewards[run][step] = obs_td['next', 'reward'].item()

        obs_td = env.step_mdp(obs_td)
