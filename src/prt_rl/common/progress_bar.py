from tqdm import tqdm


class ProgressBar:
    """
    Training Progress Bar

    Args:
        total_frames (int): Total number of frames collected

    """
    def __init__(self, total_frames):
        self.pbar = tqdm(total=total_frames)
        self.prev_iteration = 0

    def update(self, current_step: int, desc: str) -> None:
        self.pbar.set_description(desc, refresh=False)
        self.pbar.update(n=current_step - self.prev_iteration)
        self.prev_iteration = current_step