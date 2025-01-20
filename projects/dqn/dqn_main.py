import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import statistics
from tqdm import tqdm
from prt_sim.jhu.gold_explorer import GoldExplorer
from dqn import DQN


class LinearScheduler:
    def __init__(self,
                 max_epsilon,
                 min_epsilon,
                 num_episodes,
                 ):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.num_episodes = num_episodes
        self.rate = -(self.max_epsilon - self.min_epsilon) / self.num_episodes

    def get_epsilon(self, episode_number):
        eps = episode_number * self.rate + self.max_epsilon
        return eps


class ExponentialScheduler:
    def __init__(self,
                 max_epsilon,
                 min_epsilon,
                 decay_rate,
                 ):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def get_epsilon(self, episode_number):
        eps = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode_number)
        return eps


def main():
    num_episodes = 50_000
    explore_starts = False
    alpha = 5e-4
    gamma = 0.99
    max_epsilon = 1.0
    min_epsilon = 0.05
    scheduler = 'linear'
    save = False

    env = GoldExplorer()

    agent = DQN(
        state_size=1,
        num_actions=env.get_number_of_actions(),
        epsilon=max_epsilon,
        gamma=gamma,
        alpha=alpha,
        min_buffer_size=100,
        mini_batch_size=32,
        target_update_steps=250
    )

    if scheduler == 'linear':
        epsilon_scheduler = LinearScheduler(
            max_epsilon=max_epsilon,
            min_epsilon=min_epsilon,
            num_episodes=num_episodes,
        )
    elif scheduler == 'exponential':
        epsilon_scheduler = ExponentialScheduler(
            max_epsilon=max_epsilon,
            min_epsilon=min_epsilon,
            decay_rate=0.05,
        )
    else:
        eps = max_epsilon

    mean_episode_rewards = []
    mean_win_percent = []
    mean_eps_returns = deque([], maxlen=30)
    win_percent = deque([], maxlen=30)

    epsilons = np.zeros(num_episodes)
    pbar = tqdm(total=num_episodes)
    for i in range(num_episodes):
        returns = []
        wins = []

        if scheduler is not None:
            eps = epsilon_scheduler.get_epsilon(i)


        if scheduler is not None:
            agent.set_epsilon(eps)

        state = env.reset(explore_starts)
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.execute_action(action)
            trajectory = (state, action, reward, next_state, done)
            # experiences.append(trajectory)
            agent.learn(trajectory)
            state = next_state
            episode_reward += reward

        returns.append(episode_reward)

        if next_state == 126:
            wins.append(1)
        else:
            wins.append(0)

        # Agent learning
        # agent.learn(experiences)

        epsilons[i] = eps
        mean_eps_returns.append(np.mean(returns))
        win_percent.append(np.mean(wins))

        mean_episode_rewards.append(statistics.mean(mean_eps_returns))
        mean_win_percent.append(statistics.mean(win_percent))

        pbar.update(1)
        pbar.set_postfix(status=f"Episode reward: {mean_episode_rewards[i]:.2f}")

    if save:
        np.savez(f"dqn_{alpha}_{gamma}_{max_epsilon}_{scheduler}_{explore_starts}.npz",
                 epsilons=epsilons,
                 mean_episode_rewards=mean_episode_rewards,
                 mean_win_percent=mean_win_percent,
                 )

    fig, ax = plt.subplots()
    ax.plot(mean_episode_rewards)
    ax.set_title("Mean Episode Rewards")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Episode Rewards")
    ax.axhline(34, 0, num_episodes)

    plt.figure()
    plt.plot(mean_win_percent)
    plt.title("Mean Win Percentage")
    plt.xlabel("Episodes")
    plt.ylabel("Win Percentage")

    plt.figure()
    plt.plot(epsilons)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Schedule')
    plt.show()

if __name__ == '__main__':
    main()