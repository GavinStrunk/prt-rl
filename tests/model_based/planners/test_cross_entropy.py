import pytest
import torch
from torch.distributions import Normal
from prt_rl.model_based.planners.cross_entropy import CrossEntropyMethodPlanner, temporal_smooth, TanhSquashBound, ClipBound


# Temporal Smoothing Tests
def test_temporal_smooth_none_returns_tensor():
    x = torch.randn((10, 5, 2))
    smoothed = temporal_smooth(x, None)
    assert torch.equal(x, smoothed)

def test_temporal_smooth_ou():
    x = torch.tensor([[0.1], [0.2], [0.3]]).unsqueeze(0)  # Shape (1, 3, 1)
    assert x.shape == (1, 3, 1)

    # Hand calculated OU smoothing with rho=0.5
    expected = torch.tensor([[0.1], [0.15], [0.225]]).unsqueeze(0)
    assert expected.shape == (1, 3, 1)

    rho = 0.5
    smoothed = temporal_smooth(x, method='ou', rho=rho)
    assert smoothed.shape == x.shape
    assert torch.allclose(expected, smoothed)  

def test_conv_three_points_k3_hand_calc():
    """
    Simple hand-check for the convolution path.

    We use:
      x = [[[1.0],
            [2.0],
            [4.0]]]   # shape (N=1, H=3, dA=1)

    kernel_size = 3
    In your implementation:
      t = [-1, 0, 1]
      ker_un = exp(-0.5 * (t / (0.25*k))^2)   with k=3 and replicate padding.
      ker is normalized to sum to 1.

    Replicate padding for H=3, k=3 (pad=1 on each side) gives the padded signal:
      [ x0, x0, x1, x2, x2 ] = [1, 1, 2, 4, 4]

    Sliding windows (length 3):
      i=0: [1, 1, 2]
      i=1: [1, 2, 4]
      i=2: [2, 4, 4]

    Output:
      y0 = ker[0]*1 + ker[1]*1 + ker[2]*2
      y1 = ker[0]*1 + ker[1]*2 + ker[2]*4
      y2 = ker[0]*2 + ker[1]*4 + ker[2]*4

    Numerically (approx):
      ker ≈ [0.2258, 0.5484, 0.2258]
      y ≈ [1.2258, 2.2258, 3.5484]
    """
    x = torch.tensor([[[1.0], [2.0], [4.0]]], dtype=torch.float64)  # (1,3,1)

    # Run function under test
    y = temporal_smooth(x, method="conv", kernel_size=3)
    assert y.shape == x.shape

    # --- Manual hand calculation matching the implementation ---

    # Build the same kernel as in temporal_smooth
    k = 3
    t = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    ker = torch.exp(-0.5 * (t / (0.25 * k)) ** 2)
    ker = ker / ker.sum()  # normalize to sum to 1

    # Replicate padding by 1 on each side: [1, 1, 2, 4, 4]
    pad = torch.tensor([1.0, 1.0, 2.0, 4.0, 4.0], dtype=torch.float64)

    # Windows and dot products
    w0 = pad[0:3]; y0 = (w0 * ker).sum()
    w1 = pad[1:4]; y1 = (w1 * ker).sum()
    w2 = pad[2:5]; y2 = (w2 * ker).sum()

    expected = torch.tensor([[[y0], [y1], [y2]]], dtype=torch.float64)

    # Exact numerical match to manual conv
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)

    # Also check against the approximate hand numbers (gives a human-readable sanity check)
    approx = torch.tensor([[[1.2258], [2.2258], [3.5484]]], dtype=torch.float64)
    torch.testing.assert_close(y, approx, rtol=0.0, atol=5e-4)

