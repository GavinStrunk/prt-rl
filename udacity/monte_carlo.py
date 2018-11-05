import numpy as np
from collections import defaultdict
import sys


def first_visit_mc_prediction(env, num_episodes, generate_episode, gamma=1.0):
    N = np.zeros(env.action_space.n)
    returns_sums = np.zeros(env.action_space.n)


def every_visit_mc_prediction(env, num_episodes, generate_episode, gamma=1.0):
    returns_sums = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for i_episode in range(num_episodes):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = generate_episode(env)
        states, actions, rewards = zip(*episode)

        gamma_array = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            returns_sums[state][actions[i]] += sum(gamma_array[:-(i + 1)] * rewards[i:])
            N[state][actions[i]] += 1
            Q[state][actions[i]] = returns_sums[state][actions[i]] / N[state][actions[i]]
    return Q


def every_visit_mc_control(env, generate_episode, update_Q, num_episodes, alpha, gamma=1.0, eps_start=1.0,
                           eps_decay=.9999, eps_min=0.05):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        epsilon = max(epsilon * eps_decay, eps_min)
        episode = generate_episode(env, Q, epsilon, nA)
        Q = update_Q(episode, Q, alpha, gamma)

    policy = dict((k, np.argmax(v)) for k,v in Q.items())

    return policy, Q
