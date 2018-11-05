from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import sys


def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))


def epsilon_greedy_probs(env, Q_s, i_eps, eps=None):
    epsilon = 1.0 / i_eps
    if eps is not None:
        epsilon = eps
    policy_s = np.ones(env.nA) * epsilon / env.nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)

    return policy_s

def plot_performance(num_episodes, scores, plot_every):
    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))


def sarsa(env, num_episodes, alpha, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.nA))

    plot_every = 100
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)

    for i_episode in range(1, num_episodes+1):

        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        score = 0
        state = env.reset()
        policy_s = epsilon_greedy_probs(env, Q[state], i_episode)
        action = np.random.choice(np.arange(env.nA), p=policy_s)

        for t_step in np.arange(300):
            next_state, reward, done, info = env.step(action)
            score += reward

            if not done:
                policy_s = epsilon_greedy_probs(env, Q[next_state], i_episode)
                next_action = np.random.choice(np.arange(env.nA), p=policy_s)
                Q[state][action] = update_Q(Q[state][action], Q[next_state][next_action], reward, alpha, gamma)
                state = next_state
                action = next_action
            if done:
                Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
                tmp_scores.append(score)
                break

        if i_episode % plot_every == 0:
            scores.append(np.mean(tmp_scores))

    plot_performance(num_episodes, scores, plot_every)
    return Q


def q_learning(env, num_episodes, alpha, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.nA))

    plot_every = 100
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)

    for i_episode in range(1, num_episodes+1):

        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        score = 0
        state = env.reset()
        while True:
            policy_s = epsilon_greedy_probs(env, Q[state], i_episode)
            action = np.random.choice(np.arange(env.nA), p=policy_s)
            next_state, reward, done, info = env.step(action)
            score += reward

            Q[state][action] = update_Q(Q[state][action], np.max(Q[next_state]), reward, alpha, gamma)

            state = next_state

            if done:
                tmp_scores.append(score)
                break

        if i_episode % plot_every == 0:
            scores.append(np.mean(tmp_scores))

    plot_performance(num_episodes, scores, plot_every)
    return Q


def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.nA))

    plot_every = 100
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)

    for i_episode in range(1, num_episodes+1):

        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        score = 0
        state = env.reset()
        policy_s = epsilon_greedy_probs(env, Q[state], i_episode, 0.005)
        while True:
            action = np.random.choice(np.arange(env.nA), p=policy_s)
            next_state, reward, done, info = env.step(action)
            score += reward
            policy_s = epsilon_greedy_probs(env, Q[next_state], i_episode, 0.005)

            Q[state][action] = update_Q(Q[state][action], np.dot(policy_s, Q[next_state]), reward, alpha, gamma)

            state = next_state

            if done:
                tmp_scores.append(score)
                break

        if i_episode % plot_every == 0:
            scores.append(np.mean(tmp_scores))

    plot_performance(num_episodes, scores, plot_every)
    return Q
