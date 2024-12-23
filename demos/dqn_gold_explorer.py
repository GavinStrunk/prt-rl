from prt_sim.jhu.gold_explorer import GoldExplorer
from prt_rl.env.wrappers import JhuWrapper
from prt_rl.dqn import DQN


env = JhuWrapper(environment=GoldExplorer())
trainer = DQN(env)
