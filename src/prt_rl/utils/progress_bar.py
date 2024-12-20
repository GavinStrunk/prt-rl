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

    def update(self, reward_mean):
        self.pbar.set_description(f"episode_reward_mean = {reward_mean}", refresh=False)
        self.pbar.update()