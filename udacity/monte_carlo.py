import numpy as np

def first_visit_mc_prediction(env, policy, num_episodes):
    N = np.zeros(env.nS)

    returns_sums = np.zeros(env.nS)
    for i in range(num_episodes):
        episode = generate_episode(env, policy)


def every_visit_mc_prediction(env, policy, num_episodes):
    N = np.zeros(env.nS)

    returns_sums = np.zeros(env.nS)
    for i in range(num_episodes):
        episode = generate_episode(env, policy)


def generate_episode(env, policy):
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
