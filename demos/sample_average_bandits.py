r"""
======================
Sample Average Example
======================
In this example, we show how to solve the K-arm Bandits problem with Sample Averaging.

"""
from prt_rl.exact.sample_average import SampleAverage
from prt_rl.env.wrappers import JhuWrapper
from prt_sim.jhu.bandits import KArmBandits
import prt_sim.jhu.plot as pplt

# Set the number of runs and episodes per run
num_runs = 2000
num_episodes = 1000

# Initialize Environment
env = JhuWrapper(environment=KArmBandits())

# Initialize Trainer
trainer = SampleAverage(
    env=env,
)

trainer.train(num_episodes=num_episodes)

# plt.figure()
# pplt.plot_bandit_rewards(rewards)
#
# plt.figure()
# pplt.plot_bandit_percent_optimal_action(optimal_bandit, actions)
# plt.show()