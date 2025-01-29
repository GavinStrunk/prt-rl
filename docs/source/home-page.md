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
    x(y) &= 5\sum{y} \\
    y &\geq 0
\end{align}
```


# Notation