import numpy as np

'''
Requires: 
    env - nS
    step_dynamics - returns prob, next_state, reward, done 
'''
def policy_evaluation(env, step_dynamics, policy=None, gamma=1, theta=1e-8):
    value_function = np.zeros(env.nS)

    if policy is None:
        policy = create_equiprobable_policy(env.nS, env.nA)

    while True:
        delta = 0

        for state in range(env.nS):
            old_value = value_function[state]
            Vs = 0
            Q = q_from_v(env, step_dynamics, value_function, gamma)

            for action, action_prob in enumerate(policy[state]):
                Vs += action_prob * Q[state][action]

            value_function[state] = Vs
            delta = max(delta, abs(value_function[state] - old_value))
        if delta < theta:
            break

    return value_function


def create_equiprobable_policy(number_states, number_actions):
    return np.ones([number_states, number_actions]) / number_actions

def q_from_v(env, step_dynamics, value_function, gamma=1):
    Q = np.zeros([env.nS, env.nA])

    for state in range(env.nS):
        for action in range(env.nA):
            value = 0
            for prob, next_state, reward, done in step_dynamics(env, state, action):
                value += prob * (reward + gamma * value_function[next_state])
            Q[state][action] = value
    return Q


def policy_evalutation_truncated(env, step_dynamics, policy, V, max_iterations, gamma=1):

    iteration_count = 0

    while iteration_count < max_iterations:

        for state in range(env.nS):
            Vs = 0
            Q = q_from_v(env, step_dynamics, V, gamma)

            for action, action_prob in enumerate(policy[state]):
                Vs += action_prob * Q[state][action]

            V[state] = Vs
        iteration_count += 1

    return V

def policy_improvement(env, step_dynamics, value_function, gamma=1, stochastic=False):
    policy = np.zeros([env.nS, env.nA])

    for state in range(env.nS):
        Q = q_from_v(env, step_dynamics, value_function, gamma)
        if stochastic:
            policy[state] = make_stochastic_policy(Q[state])
        else:
            policy[state] = make_discrete_policy(Q[state])

    return policy


def make_stochastic_policy(qs):
    policy_step = np.zeros(qs.shape[0])

    best_actions = np.argwhere(qs == np.amax(qs))
    for a_idx in best_actions:
        policy_step[a_idx] = 1 / best_actions.shape[0]

    return policy_step


def make_discrete_policy(qs):
    policy_step = np.zeros(qs.shape)

    action_index = np.argmax(qs)
    policy_step[action_index] = 1

    return policy_step


def policy_iteration(env, step_dynamics, gamma=1, theta=1e-8, stochastic=False):
    policy = create_equiprobable_policy(env.nS, env.nA)
    policy_stable = False

    while not policy_stable:
        V = policy_evaluation(env, step_dynamics, policy, gamma, theta)
        new_policy = policy_improvement(env, step_dynamics, V, gamma, stochastic)

        if (new_policy == policy).all():
            policy_stable = True

        policy = np.copy(new_policy)
    return policy, V


def policy_iteration_truncated(env, step_dynamics, gamma=1, max_iterations=10, theta=1e-8):

    V = np.zeros([env.nS])
    policy = create_equiprobable_policy(env.nS, env.nA)

    policy_stable = False

    while not policy_stable:
        policy = policy_improvement(env, step_dynamics, V, gamma)
        old_V = np.copy(V)
        V = policy_evalutation_truncated(env, step_dynamics, policy, V, max_iterations, gamma)

        delta = max(abs(V - old_V))
        if delta < theta:
            policy_stable = True

    return policy, V


def value_iteration(env, step_dynamics, gamma=1, theta=1e-8):
    value_function = np.zeros(env.nS)
    policy_stable = False

    while not policy_stable:
        delta = 0

        for state in range(env.nS):
            old_value_function = value_function[state]

            Q = q_from_v(env, step_dynamics, value_function, gamma)
            value_function[state] = max(Q[state])

            delta = max(delta, abs(value_function[state] - old_value_function))

        if delta < theta:
            policy_stable = True

    policy = policy_improvement(env, step_dynamics, value_function, gamma)
    return policy, value_function