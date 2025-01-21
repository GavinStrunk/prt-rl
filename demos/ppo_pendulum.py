# import gymnasium as gym
# import torch
import torch.nn as nn
# from torch.distributions import Normal, Categorical
# from tqdm import tqdm
#
#
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # Define the log standard deviation
        # log_std = -0.5 * torch.ones(action_dim, dtype=torch.float32)
        # self.log_std = torch.nn.Parameter(log_std)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, actions=None):
        return self.policy(x)
        # logits = self.policy(x)
        # std = torch.exp(self.log_std)
        # dist = Normal(mu, std)
        # dist = Categorical(logits)
        # if actions is None:
        #     actions = dist.sample().unsqueeze(0)
        # log_prob_actions = dist.log_prob(actions)
        #
        # return actions, log_prob_actions.unsqueeze(0)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.critic(x)
#
# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#         self.common = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh()
#         )
#         self.actor = nn.Sequential(
#             nn.Linear(64, action_dim),
#             nn.Softmax(dim=-1)
#         )
#         self.critic = nn.Linear(64, 1)
#
#     def forward(self, x):
#         common_out = self.common(x)
#         return self.actor(common_out), self.critic(common_out).squeeze(-1)
#
# def compute_gae(rewards, values, dones, gamma, lambd):
#     """
#     Compute Generalized Advantage Estimation (GAE).
#
#     Args:
#         rewards (torch.Tensor): Rewards received at each time step (shape: [T, batch_size]).
#         values (torch.Tensor): Value function estimates (shape: [T+1, batch_size]).
#         dones (torch.Tensor): Done flags indicating the end of an episode (shape: [T, batch_size]).
#         gamma (float): Discount factor.
#         lambd (float): GAE lambda parameter.
#
#     Returns:
#         torch.Tensor: Computed GAE advantages (shape: [T, batch_size]).
#     """
#     T = rewards.size(0)
#     batch_size = rewards.size(1)
#     # Append a 0 to the values
#     values = torch.cat((values, torch.zeros(1, 1)), dim=0)
#     advantages = torch.zeros_like(rewards)  # Initialize advantages
#     gae = torch.zeros(batch_size)  # Initialize GAE buffer for batch
#
#     for t in reversed(range(T)):
#         delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
#         gae = delta + gamma * lambd * (1 - dones[t]) * gae
#         advantages[t] = gae
#
#     return advantages
#
#
# def compute_rewards_to_go(rewards, dones, gamma):
#     """
#     Compute rewards-to-go for a trajectory of rewards.
#
#     Args:
#         rewards (torch.Tensor): A 1D tensor of shape (T,) containing rewards for a trajectory.
#         gamma (float): Discount factor, between 0 and 1.
#
#     Returns:
#         torch.Tensor: A 1D tensor of shape (T,) containing the rewards-to-go for each time step.
#     """
#     returns = []
#     R = 0
#     for r, done in zip(reversed(rewards), reversed(dones)):  # Traverse the trajectory in reverse
#         if done:
#             R = 0
#         R = r + gamma * R
#         returns.insert(0, R)
#
#     return torch.tensor(returns, dtype=torch.float32)
#
#
# def compute_actor_loss(actor, obs, actions, logp_old, advantages, clip_ratio):
#     _, logp = actor(obs, actions)
#     ratio = torch.exp(logp - logp_old)
#     clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
#     loss_actor = -(torch.min(ratio * advantages, clip_adv)).mean()
#
#     approx_kl = (logp_old - logp).mean()
#     return loss_actor, approx_kl
#
#
# def compute_critic_loss(critic, obs, returns):
#     vvals = critic(obs)
#     return ((vvals - returns) ** 2).mean()
#
#
# # Parameters
# num_episodes = 1000
# batch_size = 32
# num_optimization_steps = 4
# learning_rate = 3e-4
# target_kl = 0
# epsilon = 0.2
#
# env = gym.make('CartPole-v1')
# action_dim = env.action_space.n
# state_dim = env.observation_space.shape[0]
# print(f"Action dimension: {action_dim} and state dimension: {state_dim}")
#
# # actor = Actor(state_dim, action_dim)
# # critic = Critic(state_dim)
# # actor_opt = torch.optim.Adam(actor.parameters(), lr=learning_rate)
# # critic_opt = torch.optim.Adam(critic.parameters(), lr=learning_rate)
# actor_critic = ActorCritic(state_dim, action_dim)
# optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)
#
# pbar = tqdm(total=num_episodes, desc="Episode Reward: 0")
# total_frames = 0
# for i in range(num_episodes):
#     # Collect experience using the current policy (actor)
#     states = []
#     actions = []
#     log_probs = []
#     v_vals = []
#     rewards = []
#     episode_reward = 0
#     next_states = []
#     dones = []
#
#     obs, _ = env.reset()
#     obs = torch.from_numpy(obs).float()
#     done = False
#     while not done:
#         # action, log_prob_actions = actor(obs)
#         # vval = critic(obs)
#         probs, values = actor_critic(obs)
#         dist = Categorical(probs)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#
#         next_obs, reward, term, trunc, _ = env.step(action.item())
#         done = term or trunc
#         episode_reward += reward
#
#         # Save experience
#         states.append(obs.unsqueeze(0))
#         actions.append(action.unsqueeze(0))
#         log_probs.append(log_prob.unsqueeze(0))
#         v_vals.append(values.unsqueeze(0))
#         next_states.append(torch.from_numpy(next_obs).float().unsqueeze(0))
#         rewards.append(torch.tensor([reward]).float().unsqueeze(0))
#         dones.append(torch.tensor([done]).float().unsqueeze(0))
#
#         # Update the observation - i.e. step the MDP
#         obs = torch.from_numpy(next_obs).float()
#
#     states = torch.cat(states, dim=0)
#     actions = torch.cat(actions, dim=0)
#     log_probs = torch.cat(log_probs, dim=0)
#     v_vals = torch.cat(v_vals, dim=0)
#     next_states = torch.cat(next_states, dim=0)
#     rewards = torch.cat(rewards, dim=0)
#     dones = torch.cat(dones, dim=0)
#
#     # advantages = compute_gae(rewards, v_vals, dones, gamma=0.99, lambd=0.95)
#     ret = compute_rewards_to_go(rewards, dones, gamma=0.99)
#     advantages = (ret - v_vals.squeeze().detach())
#
#     # Optimize the actor and critic
#     log_probs = log_probs.detach()
#     advantages = advantages.detach()
#     for _ in range(num_optimization_steps):
#         # actor_opt.zero_grad()
#         # actor_loss, kl = compute_actor_loss(actor, states, actions, log_probs, advantages,
#         #                                     clip_ratio=0.2)
#         #
#         # # if kl.item() > 1.5 * target_kl:
#         # #     break
#         # actor_loss.backward()
#         # # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
#         # actor_opt.step()
#         #
#         # critic_opt.zero_grad()
#         # critic_loss = compute_critic_loss(critic, states, ret)
#         # # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
#         # critic_loss.backward()
#         # critic_opt.step()
#         for i in range(0, len(states), batch_size):
#             batch_states = states[i:i + batch_size]
#             batch_actions = actions[i:i + batch_size]
#             batch_log_probs_old = log_probs[i:i + batch_size]
#             batch_advantages = advantages[i:i + batch_size]
#             batch_returns = ret[i:i + batch_size]
#
#             new_probs, new_vals = actor_critic(batch_states)
#             dist = Categorical(new_probs)
#             new_log_probs = dist.log_prob(batch_actions)
#
#             # Policy Loss
#             ratio = torch.exp(new_log_probs - batch_log_probs_old)
#             clipped_adv = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
#             policy_loss = -torch.min(ratio * batch_advantages, clipped_adv).mean()
#
#             # Value Loss
#             values_loss = nn.functional.mse_loss(new_vals.squeeze(), batch_returns)
#
#             # Entropy Loss
#             entropy = dist.entropy().mean()
#
#             total_loss = policy_loss + 0.5*values_loss - 0.01 * entropy
#
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#
#
#             # actor_loss, kl = compute_actor_loss(actor, batch_states, batch_actions, batch_log_probs_old, batch_advantages, clip_ratio=0.2)
#             #
#             # # if kl.item() > 1.5 * target_kl:
#             # #     break
#             # actor_loss.backward()
#             # # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
#             # actor_opt.step()
#             #
#             # critic_opt.zero_grad()
#             # critic_loss = compute_critic_loss(critic, batch_states, batch_returns)
#             # # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
#             # critic_loss.backward()
#             # critic_opt.step()
#
#     pbar.set_description(f"Episode Reward: {episode_reward}", refresh=False)
#     pbar.update()
#
#
# # Evaluate Policy
# eval_env = gym.make('CartPole-v1', render_mode='human')
# obs, _ = eval_env.reset()
# obs = torch.from_numpy(obs).float()
# done = False
# eval_reward = 0
#
# while not done:
#     probs, _ = actor_critic(obs)
#     dist = Categorical(probs)
#     action = dist.sample()
#     next_obs, reward, term, trunc, _ = eval_env.step(action.item())
#     done = term or trunc
#     eval_reward += reward
#
# print(f'Episode Reward: {eval_reward}')

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        common_out = self.common(x)
        return self.actor(common_out), self.critic(common_out)


def compute_returns(rewards, dones, gamma):
    """
    Compute rewards-to-go for each timestep in an episode.
    """
    returns = []
    R = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def train_ppo(env, model, optimizer, gamma, epsilon, num_epochs, batch_size):
    """
    Train the PPO algorithm.
    """
    state = env.reset()[0]
    done = False

    # Storage for trajectories
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    # Collect rollout
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # probs, value = model(state_tensor)
        probs = actor(state_tensor)
        value = critic(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()

        next_state, reward, term, trunc, _ = env.step(action.item())
        done = term or trunc

        # Store transition
        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(dist.log_prob(action))
        values.append(value)

        state = next_state

    # Convert collected data to tensors
    states = torch.stack(states).detach()
    actions = torch.stack(actions).detach()
    log_probs = torch.stack(log_probs).detach()
    values = torch.stack(values).squeeze().detach()
    returns = compute_returns(rewards, dones, gamma)
    returns = returns.detach()
    advantages = (returns - values.detach())
    advantages = advantages.detach()

    # PPO Updates
    for _ in range(num_epochs):
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            batch_actions = actions[i:i + batch_size]
            batch_log_probs = log_probs[i:i + batch_size]
            batch_advantages = advantages[i:i + batch_size]
            batch_returns = returns[i:i + batch_size]

            # Recompute log probs and value estimates
            # new_probs, new_values = model(batch_states)
            new_probs = actor(batch_states)
            new_values = critic(batch_states)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(batch_actions)

            # Policy loss
            ratio = torch.exp(new_log_probs - batch_log_probs)
            clipped_adv = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * batch_advantages
            policy_loss = -torch.min(ratio * batch_advantages, clipped_adv).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(new_values.squeeze(), batch_returns)

            # Entropy bonus
            entropy = dist.entropy().mean()
            # loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            # Backward pass
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            value_loss.backward()
            critic_optimizer.step()

    # for _ in range(num_epochs):
    #     batch_states = states
    #     batch_actions = actions
    #     batch_log_probs = log_probs
    #     batch_advantages = advantages
    #     batch_returns = returns
    #
    #     # Recompute log probs and value estimates
    #     new_probs, new_values = model(batch_states)
    #     dist = Categorical(new_probs)
    #     new_log_probs = dist.log_prob(batch_actions)
    #
    #     # Policy loss
    #     ratio = torch.exp(new_log_probs - batch_log_probs)
    #     clipped_adv = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * batch_advantages
    #     policy_loss = -torch.min(ratio * batch_advantages, clipped_adv).mean()
    #
    #     # Value loss
    #     value_loss = nn.functional.mse_loss(new_values.squeeze(), batch_returns)
    #
    #     # Entropy bonus
    #     entropy = dist.entropy().mean()
    #     loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    #
    #     # Backward pass
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    return sum(rewards)  # Return total reward for this episode


# Environment and Hyperparameters
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

# PPO Parameters
gamma = 0.99
epsilon = 0.1
num_epochs = 10
batch_size = 32

# Training Loop
for episode in range(1000):
    total_reward = train_ppo(env, model, optimizer, gamma, epsilon, num_epochs, batch_size)
    print(f"Episode {episode}: Total Reward = {total_reward}")
    if total_reward >= 500:  # Solved condition for CartPole-v1
        print("Solved!")
        break
