"""
Adaptive and meta-learning RL algorithms.

This module contains agents that are designed to generalize and adapt
to new dynamics, tasks, or environments with minimal additional data.
These include:
  - rapid adaptation / online system-identification methods
    (e.g. policies conditioned on inferred environment embeddings),
  - meta-RL style methods that learn to learn across task families
    (e.g. context encoders, fast fine-tuning, recurrent adaptation).

In contrast to standard model-free algorithms that learn a fixed policy
for a single MDP, these agents focus on fast test-time adaptation,
sim-to-real robustness, and cross-task generalization.
"""