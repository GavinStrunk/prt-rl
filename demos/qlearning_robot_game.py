from prt_sim.jhu.robot_game import RobotGame
from prt_rl.env.wrappers import JhuWrapper
from prt_rl.exact.qlearning import QLearning


env = JhuWrapper(environment=RobotGame())

trainer = QLearning(env)
trainer.train()