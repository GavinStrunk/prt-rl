import torch
from prt_rl.common.metrics import MetricTracker, TensorDeque

def test_initializing_metric_callbacks():
    def post_episode_callback(self, reward, info) -> None:
        self.update_metric('reward', reward)

    tracker = MetricTracker(
        post_episode_callback=post_episode_callback
    )

    tracker.post_episode(rewards=torch.tensor([[5]]), info={})
    print(tracker.metrics)

def test_metric_tracking():
    tracker = MetricTracker()
    val = torch.tensor([1.0])
    val2 = torch.tensor([2.0])
    val3 = torch.tensor([3.0])

    tracker.update_metric('queue_metric', val)
    assert tracker.metrics['queue_metric'] == torch.tensor([[1.0]])
    tracker.update_metric('queue_metric', val2)
    assert tracker.metrics['queue_metric'].shape == (2, 1)
    assert torch.allclose(tracker.metrics['queue_metric'], torch.tensor([[1.0], [2.0]]))
    tracker.update_metric('queue_metric', val3)
    assert tracker.metrics['queue_metric'].shape == (3, 1)
    assert torch.allclose(tracker.metrics['queue_metric'], torch.tensor([[1.0], [2.0], [3.0]]))

def test_configured_metric():
    tracker = MetricTracker()
    tracker.configure_metric('queue_metric', queue_length=2)

    val = torch.tensor([1.0])
    val2 = torch.tensor([2.0])
    val3 = torch.tensor([3.0])

    tracker.update_metric('queue_metric', val)
    assert tracker.metrics['queue_metric'] == torch.tensor([[1.0]])
    tracker.update_metric('queue_metric', val2)
    assert tracker.metrics['queue_metric'].shape == (2, 1)
    assert torch.allclose(tracker.metrics['queue_metric'], torch.tensor([[1.0], [1.5]]))
    tracker.update_metric('queue_metric', val3)
    assert tracker.metrics['queue_metric'].shape == (3, 1)
    assert torch.allclose(tracker.metrics['queue_metric'], torch.tensor([[1.0], [1.5], [2.5]]))


def test_tensor_deque():
    queue = TensorDeque(maxlen=2)
    val = torch.tensor([1.0])
    val2 = torch.tensor([2.0])
    val3 = torch.tensor([3.0])
    assert val.shape == (1,)

    # Add to queue
    queue.append(val)
    assert queue.buffer.shape == (1, 1)
    queue.append(val2)
    assert queue.buffer.shape == (2, 1)
    queue.append(val3)
    assert queue.buffer.shape == (2, 1)
    assert torch.allclose(queue.get(), torch.tensor([[2.0], [3.0]]))

