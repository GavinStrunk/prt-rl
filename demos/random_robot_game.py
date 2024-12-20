from prt_rl.utils.policies import RandomPolicy
from prt_rl.env.wrappers import JhuWrapper
from prt_sim.jhu.robot_game import RobotGame

env = JhuWrapper(environment=RobotGame())
policy = RandomPolicy(env_params=env.get_parameters())

state_td = env.reset()
done = False
while not done:
    state_td = env.step(policy.get_action(state_td))
    done = state_td['next', 'done']
