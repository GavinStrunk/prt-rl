import inspect
from tensordict.tensordict import TensorDict
import torch
from typing import Callable, Optional


class MetricTracker:
    """
    The metric tracker stores metrics over the entire training period so plots can be generated based on a time series of values. This is different from the Logger because it does not just record the metrics, but allows for custom post-processing after training has completed.

    Args:
        pre_episode_callback (Callable, optional): Callback that is called after the environment is reset, but before any steps are taken.
        post_episode_callback (Callable, optional): Callback that is called after the episode ends.
        device (str): Device the metrics are stored on. Defaults to 'cpu'.

    Examples:
        # Registering a post episode callback
        def post_episode_callback(self, reward, info):
    """

    def __init__(self,
                 pre_episode_callback: Optional[Callable] = None,
                 post_episode_callback: Optional[Callable] = None,
                 device: str = "cpu"
                 ) -> None:
        """
        Initialize the MetricTracker.

        Args:
            device (str): Device to store metrics ("cpu" or "cuda").
            post_episode_callback (Callable): Callback function to compute metrics post-episode.
        """
        self.metrics = TensorDict({}, batch_size=[]).to(device)
        self.queues = TensorDict({}, batch_size=[]).to(device)
        self.pre_episode_callback = pre_episode_callback
        self.post_episode_callback = post_episode_callback
        self.device = device

    def _validate_callback(self, callback):
        """
        Validate that the callback matches the expected prototype.

        Args:
            callback (callable): The callback function to validate.

        Raises:
            TypeError: If the callback does not match the required signature.
        """
        sig = inspect.signature(callback)
        params = list(sig.parameters.values())
        if len(params) != 3 or params[0].name != "self" or params[1].name != "rewards" or params[2].name != "info":
            raise TypeError(
                f"Callback {callback.__name__} must have the signature (self, rewards, info)."
            )

    def configure_metric(self,
                         key: str,
                         queue_length: Optional[int] = None,
                         ) -> None:
        """
        Configures the tracking style for the metric.

        Args:
            key (str): The name of the metric.
            queue_length (Optional[int]): Creates a queued metric with a window size of queue_length. The mean of the queue values is stored in the metric at each episode.
        """
        if queue_length is not None:
            self.queues[key] = TensorDeque(maxlen=queue_length, device=self.device)

    def update_metric(self,
                      key: str,
                      value: torch.Tensor
                      ) -> None:
        """
        Update a metric, either by appending, queuing, or storing a single value.

        Args:
            key (str): The name of the metric.
            value (any): The value to update the metric with.
        """
        # Add value if this is a queued metric
        if key in self.queues.keys():
            self.queues[key].append(value)
            # Compute mean value
            value = torch.mean(self.queues[key].get(), dim=0)

        # Add a dimension to the front which is used as the # updates
        value = value.unsqueeze(0)

        # Create the metric if it does not exist
        if key not in self.metrics.keys():
            self.metrics.set(key, value)
        else:
            self.metrics.set(key, torch.cat([self.metrics[key], value], dim=0))

    def get_metric(self) -> TensorDict:
        """
        Returns the entire metrics tensordict.

        Returns:
            TensorDict: Metrics tensordict.
        """
        return self.metrics

    def pre_episode(self):
        """
        Execute the registered pre-episode callback.
        """
        if self.pre_episode_callback is not None:
            self.pre_episode_callback(self)

    def post_episode(self, rewards, info):
        """
        Execute the registered post-episode callbacks.

        Args:
            rewards: The rewards from the episode.
            info: Additional episode information.
        """
        if self.post_episode_callback is not None:
            self.post_episode_callback(self, rewards, info)


class TensorDeque:
    """
    TensorDeque is a queue for tensors. It is meant to mimic deque. The values are stored in the shape (# elements, values).

    Args:
        maxlen (int): Maximum length of the deque.
        device (str): Device to store metrics ("cpu" or "cuda"). Defaults to "cpu".
    """

    def __init__(self,
                 maxlen: int,
                 device="cpu"
                 ) -> None:
        """
        Initialize a fixed-size sliding tensor buffer.

        Args:
            maxlen (int): Maximum length of the buffer.
            device (str): Device to store the buffer ("cpu" or "cuda").
        """
        self.buffer = torch.empty(0, device=device)
        self.maxlen = maxlen

    def append(self, value):
        """
        Append a value to the buffer.

        Args:
            value (torch.Tensor or float): Value to append.
        """
        value = torch.tensor([value], device=self.buffer.device) if not torch.is_tensor(value) else value.unsqueeze(0)
        self.buffer = torch.cat([self.buffer, value])
        if len(self.buffer) > self.maxlen:
            self.buffer = self.buffer[-self.maxlen:]

    def get(self):
        """
        Get the current buffer as a tensor.

        Returns:
            torch.Tensor: The current buffer.
        """
        return self.buffer

# Example Usage
# def compute_rewards_callback(self, rewards, info):
#     # Example: Compute and log total reward from rewards tensor
#     total_reward = rewards.sum().item()
#     self.update_metric("total_reward", total_reward, mode="append")
#
#
# # Instantiate MetricTracker with callbacks
# tracker = MetricTracker(post_episode_callback=[compute_rewards_callback])
#
# # Simulate training loop
# for episode in range(5):
#     rewards = torch.rand(10)  # Example rewards for an episode
#     info = {"episode": episode}  # Example additional info
#     tracker.post_episode(rewards, info)  # Run post-episode callbacks
#     tracker.update_metric("loss", torch.rand(1).item(), mode="append")  # Update loss
#
# # Log metrics
# tracker.log_metrics()
