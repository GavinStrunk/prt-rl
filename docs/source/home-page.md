The Python Research Toolkit: Reinforcement Learning or prt_rl for short is a collection of reinforcement learning
algorithm implemented with the purpose of exploring the fundamental mathematics of RL. The emphasis of these algorithms
is readability and exploring the equations and tips and tricks. The goal is not to provide the highest performance.
There are other libraries that achieve high performance like TorchRL, RLlib, Tianshou, etc. Therefore, this repository
is for learning and academic purposes.

Also included as a separate package in this package is Python Research Toolkit - Simulation or prt_sim. This package
includes a port of simulation environments used in the RL course at Johns Hopkins, as well as, a rendering class for
grid world environments. Discrete worlds can be built based on this package, but it is really intended as smaller and
simpler discrete test cases for RL algorithms. If a custom environment is created, regardless of the simulation package
it is built on, it should be done in this package.

# What is Reinforcement Learning

The fundamental problem we are trying to solve is:
```{math}
\begin{align}
    \theta^* &= arg \max_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ \sum_t r(s_t,a_t)\right] \\
    p_\theta(s_1, a_1, ..., s_T, a_T) &= p(s_1) \prod_{t=1}^T \pi_\theta(a_t | s_t) p(s_{t+1}| s_t, a_t)
\end{align}
```
