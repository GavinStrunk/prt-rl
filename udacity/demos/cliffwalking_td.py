import gym
from cliffwalking_plot_utils import *
from temporal_difference import *


def print_optimal_solution():
    V_opt = np.zeros((4, 12))
    V_opt[0:13][0] = -np.arange(3, 15)[::-1]
    V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
    V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
    V_opt[3][0] = -13

    plot_values(V_opt)


def run_sarsa(env):
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsa = sarsa(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4, 12)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsa)

    # plot the estimated optimal state-value function
    V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)


def run_qlearning(env):
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsamax = q_learning(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsamax = np.array(
        [np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4, 12))
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsamax)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])


def run_expected_sarsa(env):
    # obtain the estimated optimal policy and corresponding action-value function
    Q_expsarsa = expected_sarsa(env, 10000, 1)

    # print the estimated optimal policy
    policy_expsarsa = np.array(
        [np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4, 12)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_expsarsa)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])


def main():
    env = gym.make('CliffWalking-v0')

    print(env.observation_space)
    print(env.action_space)

    run_sarsa(env)

    run_qlearning(env)

    run_expected_sarsa(env)


if __name__ == '__main__':
    main()