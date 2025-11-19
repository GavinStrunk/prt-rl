# Installation

This page explains how to install **prt-rl** and the optional environment libraries used by its wrappers.  
The core library is lightweight and has minimal dependencies; environments are installed separately to keep the installation fast and flexible.

## Requirements

- **Python ≥ 3.11**
- **PyTorch ≥ 2.6**

## Install prt-rl

You can install the library using either **pip** or **uv**:

**Using pip**
```bash
pip install prt-rl
```

**Using uv**
```bash
uv add prt-rl
```

## Installing Environment Dependencies
prt-rl does **not** install environment libraries by default.
This avoids unnecessary dependencies for users who only want the algorithm implementations.

Install only the environment packages you plan to use:

**Gymnasium**
```bash
uv add gymnasium
```

**VMAS (Vectorized Multi-Agent Simulator)**
```bash
uv add vmas
```

**Isaac Lab / Isaac Sim**
```bash
uv add isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
```

**PRT-SIM (Python Research Toolkit Simulation)**
```bash
uv add prt-sim
```

## Verifying Your Installation
To confirm that prt-rl is installed correctly:

```bash
python -c "import prt_rl; print('prt-rl version:', prt_rl.__version__)"
```

If you installed environment packages, check them as well:

```bash
python -c "import gymnasium, vmas"
```