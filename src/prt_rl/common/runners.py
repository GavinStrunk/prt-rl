from typing import Optional, List
from prt_rl.env.interface import EnvironmentInterface
from prt_rl.common.recorders import Recorder
from prt_rl.common.visualizers import Visualizer
from prt_rl.agent import Agent
from prt_rl.common.policies.interface import Policy


def watch(env: EnvironmentInterface, policy: Policy, num_episodes: int = 1) -> None:
    """
    Watch a trained RL agent in a gym environment.

    Args:
        env: The environment to run the agent in.
        policy: The RL policy to use for acting in the environment.
    """
    episode_rewards = []
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = policy.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print(f"Episode {i} Total reward: {total_reward.cpu().item()}")
        episode_rewards.append(total_reward.cpu().item())

    avg_reward = sum(episode_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")



class Runner:
    """
    A runner executes an agent in an environment. It simplifies the process of evaluating agents that have been trained.

    The runner assumes the rgb_array is in the info dictioanary and has shape (num_envs, channel, height, width).

    .. note::
        To use the visualizer, the environment wrapper render mode must be set to 'rgb_array'.
    
    Args:
        env (EnvironmentInterface): the environment to run the agent in
        agent (BaseAgent): Agent to be executed in the environment
        recorders (Optional[List[Recorder]]): List of recorders to record the experience and info during the run
        visualizer (Optional[Visualizer]): Visualizer to show the environment frames during the run
    """
    def __init__(self,
                 env: EnvironmentInterface,
                 agent: Agent,
                 recorders: Optional[List[Recorder]] = None,
                 visualizer: Optional[Visualizer] = None,
                 ) -> None:
        self.env = env
        self.agent = agent
        self.recorders = recorders or []
        self.visualizer = visualizer

    def run(self):
        # Reset the environment and recorder
        for r in self.recorders:
            r.reset()

        state, info = self.env.reset()
        done = False

        # Start visualizer and show initial frame
        if self.visualizer is not None:
            self.visualizer.start()
            self.visualizer.show(info['rgb_array'][0])

        for r in self.recorders:
            r.record_info(info)

        # Loop until the episode is done
        while not done:
            action = self.agent.act(state, deterministic=True)
            next_state, reward, done, info = self.env.step(action)

            # Record the environment frame
            if self.visualizer is not None:
                self.visualizer.show(info['rgb_array'][0])

            for r in self.recorders:
                r.record_info(info)
                r.record_experience({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })

            state = next_state

        if self.visualizer is not None:
            self.visualizer.stop()

        # Save the recording
        for r in self.recorders:
            r.close()

        self.env.close()
