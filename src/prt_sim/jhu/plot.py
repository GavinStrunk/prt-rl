import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from typing import List
from prt_sim.jhu.bandits import KArmBandits

def plot_bandit_probabilities(env: KArmBandits) -> None:
    """
    Plots the mean and variance of the bandit probabilities.

    Args:
        env (KArmBandits): bandits environment

    """
    probs = env.bandit_probs

    plt.errorbar(
        np.arange(len(probs)),
        probs,
        yerr=np.ones(len(probs)),
        fmt='o',
        linewidth=2,
        capsize=6,
    )

    plt.xlabel("Action")
    plt.ylabel("Reward Distribution")
    plt.title(f"{len(probs)}-armed Testbed")

def plot_bandit_rewards(rewards: np.ndarray) -> None:
    """
    Plots the rewards received by the agent(s) playing the bandits game.

    Args:
        rewards (np.ndarray): rewards received by the agent(s) with shape (# agents, # episodes)

    """
    if rewards.shape[0] == 1:
        plt.plot(rewards[0])
    else:
        # Compute mean of rewards
        means = np.mean(rewards, axis=0)
        stds = np.std(rewards, axis=0)

        # Compute confidence interval of rewards
        t_critical = t.ppf(0.975, df=rewards.shape[0] - 1)
        ci_margin = t_critical * (stds / np.sqrt(rewards.shape[0]))
        ci_upper = means + ci_margin
        ci_lower = means - ci_margin

        plt.plot(np.arange(rewards.shape[-1]), means)
        plt.fill_between(np.arange(rewards.shape[-1]), ci_lower, ci_upper, alpha=0.20)

    plt.xlabel('Steps')
    plt.ylabel('Average Rewards')
    plt.title("Average Agent Rewards")

def plot_bandit_percent_optimal_action(optimal_bandits: np.ndarray, actions: np.ndarray) -> None:
    """
    Creates a plot of the percentage of optimal actions over the training episodes.

    Args:
        optimal_bandits (np.ndarray): array of optimal bandit indexes
        actions (np.ndarray): actions chosen by the agent(s) with shape (# agents, # episodes)

    """
    # Sum the number of times the optimal action was chosen
    optimal_actions = np.sum((actions == optimal_bandits[:, np.newaxis]), axis=0).astype(float)

    # Divide the count by the number of runs in the step
    optimal_action_percent = optimal_actions / float(actions.shape[0]) * 100.0

    plt.plot(optimal_action_percent)

    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.title("Average Agent Optimal Action Selection")