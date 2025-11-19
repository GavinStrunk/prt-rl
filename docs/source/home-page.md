# **The Python Research Toolkit: Reinforcement Learning (prt-rl)**

**prt-rl** is part of the broader *Python Research Toolkit* ecosystem and provides a clean, mathematically grounded collection of reinforcement learning algorithms.  
Its primary goal is **clarity, pedagogy, and research exploration**—not raw performance.

This library is designed for researchers, students, and practitioners who want to understand *why* RL algorithms work, how their mathematics map to code, and what practical implementation details matter in real systems.

Unlike high-performance libraries such as **TorchRL**, **RLlib**, and **skrl**, **prt-rl focuses on transparency, composability, and conceptual depth**. Every algorithm is implemented with an emphasis on readability, modularity, and annotated code that highlights both the underlying equations and the implementation tips that make them work in practice.

---

## Main Contributions

- **Composable Architecture with Minimal Inheritance**  
  Designed around simple, reusable building blocks so new algorithms can be created easily.  
  Emphasis is placed on composition over inheritance to reduce architectural overhead and promote extensibility.

- **Broad Coverage of RL Paradigms**  
  Includes state-of-the-art **model-free algorithms** (e.g., PPO, SAC, DQN, TD3) as well as support for other branches of RL such as **model-based RL**, **meta-RL**, and **multi-agent RL**.

- **Mathematically Faithful Implementations**  
  Algorithms follow their original mathematical derivations closely.  
  Code is extensively commented with explanations, references to key papers, and practical “tips and tricks” used in modern RL research.

>⚠️ **Note:** This repository is under active development. APIs, file structure, and module organization may change as the project evolves. Backward compatibility is not guaranteed until version 1.0.

**Github Repository**: [https://github.com/GavinStrunk/prt-rl](https://github.com/GavinStrunk/prt-rl)

To begin using this library follow the {doc}`Installation <source/installation>` instructions and then see the {doc}`Examples <gallery>`.