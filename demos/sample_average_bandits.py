r"""
======================
Sample Average Example
======================
In this example, we show how to solve the K-arm Bandits problem with Sample Averaging.

"""
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from prt_rl.exact.sample_average import SampleAverage
from prt_rl.env.wrappers import JhuWrapper
from prt_sim.jhu.bandits import KArmBandits
import prt_sim.jhu.plot as pplt

# Set the number of runs and episodes per run
num_runs = 2000
num_episodes = 1000

# # Initialize the Bandit environment
# env = KArmBandits()
#
# # Initialize logging metrics
# actions = np.zeros((num_runs, num_episodes), dtype=int)
# rewards = np.zeros((num_runs, num_episodes))
# optimal_bandit = np.zeros(num_runs, dtype=int)
#
# # Loop for the number of runs
# for run in range(num_runs):
#
#     # Reset the environment
#     env.reset()
#
#     # Log the optimal bandit for the environment
#     optimal_bandit[run] = env.get_optimal_bandit()
#
#     # Create a Sample Average agent
#     agent = SampleAverage(num_actions=env.get_number_of_actions(), epsilon=0.1)
#
#     # Execute the environment
#     for step in range(num_episodes):
#         action = agent.select_action()
#         actions[run][step] = action
#         _, reward, _ = env.execute_action(action)
#         rewards[run][step] = reward
#         agent.learn(action, reward)

# Initialize Environment
env = JhuWrapper(
    environment=KArmBandits()
)

# Initialize Trainer
trainer = SampleAverage(
    env_params=env.get_parameters(),
    epsilon=0.1
)

trainer.train(num_episodes=num_episodes)

# plt.figure()
# pplt.plot_bandit_rewards(rewards)
#
# plt.figure()
# pplt.plot_bandit_percent_optimal_action(optimal_bandit, actions)
# plt.show()