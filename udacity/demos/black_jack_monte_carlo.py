import gym
from blackjack_plot_utils import *
from monte_carlo import *

def generate_episode(env):
    episode = []
    state = env.reset()

    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break

    return episode

def generate_episode_from_Q(env, Q, epsilon, nA):
    episode = []
    state = env.reset()

    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def get_probs(Q_s, epsilon, nA):
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s


def update_Q(episode, Q, alpha, gamma):
    states, actions, rewards = zip(*episode)

    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_q = Q[state][actions[i]]
        Q[state][actions[i]] = old_q + alpha*(sum(rewards[i:]*discounts[:-(i+1)]) - old_q)

    return Q


def main():
    env = gym.make('Blackjack-v0')

    Q = every_visit_mc_prediction(env, 500000, generate_episode)

    # obtain the corresponding state-value function
    V_to_plot = dict((k, (k[0] > 18) * (np.dot([0.8, 0.2], v)) + (k[0] <= 18) * (np.dot([0.2, 0.8], v))) \
                     for k, v in Q.items())

    # plot the state-value function
    plot_blackjack_values(V_to_plot)

    # obtain the corresponding state-value function
    V = dict((k, np.max(v)) for k, v in Q.items())

    # obtain the estimated optimal policy and action-value function
    policy, Q = every_visit_mc_control(env, generate_episode_from_Q, update_Q, 500000, 0.02)

    # obtain the corresponding state-value function
    V = dict((k, np.max(v)) for k, v in Q.items())
    # plot the state-value function
    plot_blackjack_values(V)

    # plot the policy
    plot_policy(policy)


if __name__ == '__main__':
    main()
