from tqdm import tqdm


class ProgressBar:
    """
    Training Progress Bar

    Args:
        total_frames (int): Total number of frames collected
        frames_per_batch (int): Number of frames per batch

    """
    def __init__(self, total_frames, frames_per_batch):
        self.pbar = tqdm(total=total_frames // frames_per_batch, desc="episode_reward_mean = 0")

    def update(self, epsiode_reward, cumulative_reward):
        self.pbar.set_description(f"Episode Reward: {epsiode_reward}  Cumulative Reward: {cumulative_reward}", refresh=False)
        self.pbar.update()