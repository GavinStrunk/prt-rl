from tqdm import tqdm


class ProgressBar:
    """
    Training Progress Bar

    Args:
        total_frames (int): Total number of frames collected
        frames_per_batch (int): Number of frames per batch

    """
    def __init__(self, total_frames):
        self.pbar = tqdm(total=total_frames, desc="episode_reward_mean = 0")

    def update(self, current_step, epsiode_reward, cumulative_reward):
        self.pbar.set_description(f"Episode Reward: {epsiode_reward}  Cumulative Reward: {cumulative_reward}", refresh=False)
        self.pbar.update(n=current_step)