import gym
from blackjack_plot_utils import *

def main():
    env = gym.make('Blackjack-v0')

    print(env.observation_space)
    print(env.action_space)


if __name__ == '__main__':
    main()
