import numpy as np
from frozenlake import FrozenLakeEnv
from frozenlake_plot_utils import plot_values
from dynamic_programming import *


def get_one_step_dynamics(env, state, action):
    # Return: probability, next state, reward, done
    return env.P[state][action]


def main():
    env = FrozenLakeEnv(is_slippery=True)

    print("Start Frozen Lake DP Demo")

    #V = policy_evaluation(env, get_one_step_dynamics)
    #plot_values(V)
    policy_pi, value_pi = policy_iteration(env, get_one_step_dynamics, stochastic=True)
    print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
    print(policy_pi, "\n")
    plot_values(value_pi)

    policy_tpi, V_tpi = policy_iteration_truncated(env, get_one_step_dynamics, max_iterations=2)

    # print the optimal policy
    print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
    print(policy_tpi, "\n")

    # plot the optimal state-value function
    plot_values(V_tpi)

    policy_vi, V_vi = value_iteration(env, get_one_step_dynamics)

    # print the optimal policy
    print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
    print(policy_vi, "\n")

    # plot the optimal state-value function
    plot_values(V_vi)


if __name__ == '__main__':
    main()
