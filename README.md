# BanditLab

[![PyPI version](https://img.shields.io/pypi/v/banditlab.svg)](https://pypi.org/project/banditlab/)
[![Python versions](https://img.shields.io/pypi/pyversions/banditlab.svg)](https://pypi.org/project/banditlab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Surikkat/MAB_Framework.svg?style=social)](https://github.com/Surikkat/MAB_Framework)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://surikkat.github.io/MAB_Framework/)

**A modular framework for reproducible multi-armed bandit research.**

20+ algorithms · 3 levels of access · delayed feedback · offline policy evaluation

---

## Why BanditLab

- **20+ bandit algorithms** — from classical (ε-greedy, UCB, Thompson Sampling) to state-of-the-art (NeuralUCB, GP-TS, NN-AGP, delayed feedback bandits).
- **Model–algorithm decoupling** — combine any algorithm with any reward model (linear, GLM, Gaussian process, neural network).
- **Three levels of access** — Python API for researchers, YAML configs for reproducible experiments, Streamlit GUI for non-programmers.
- **Built for research** — offline policy evaluation (DM, IPS, DR), delayed feedback support, and reproducible benchmarks.
- **Lightweight core** — `pip install banditlab` pulls only NumPy, pandas, scipy, scikit-learn, and PyYAML. Neural and GUI components are optional extras.

## Installation

```bash
pip install banditlab
```

Optional extras:

```bash
pip install banditlab[deep]          # Neural bandits (PyTorch)
pip install banditlab[optimization]  # RegCB (cvxpy)
pip install banditlab[obp]           # Open Bandit Dataset support
pip install banditlab[viz]           # Matplotlib / Seaborn / Plotly
pip install banditlab[gui]           # Streamlit GUI
pip install banditlab[all]           # Everything
```

**CPU-only PyTorch**: [deep] pulls the default (GPU) PyTorch wheel (~550 MB).
For a lighter install, pre-install the CPU build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install banditlab[deep]
```

Requirements: Python 3.10–3.12.

## Quickstart (60 seconds)
```python
from mab_framework import (
    SyntheticLinearEnv, ThompsonSampling, OnlineRidgeRegression,
)

# 1. Environment: 3 arms, 5-dimensional context
env = SyntheticLinearEnv(n_arms=3, context_dim=5)
env.reset()

# 2. Algorithm: Thompson Sampling with a linear reward model
model = OnlineRidgeRegression(feature_dim=5, l2_reg=1.0)
bandit = ThompsonSampling(n_arms=3, model=model)

# 3. Interaction loop
regret = 0.0
for _ in range(1000):
    ctx = env.get_context()
    arm = bandit.select_arm(ctx)
    step = env.step(arm, ctx)
    bandit.update(step["available_rewards"])
    regret += step["optimal_reward"] - step["instant_reward"]

print(f"Cumulative regret: {regret:.2f}")
```
More examples in [examples/]().

## Three levels of access

### 1. Python API 
Full programmatic control. See [docs/quickstart.md]() and [examples/]().

### 2. YAML configuration
Reproducible experiments without writing code:

```yaml
experiment:
  name: "synthetic_test"
  steps: 1000
  n_runs: 10

environment:
  name: "SyntheticLinearEnv"
  params: { n_arms: 3, context_dim: 5 }

algorithms:
  - name: "ThompsonSampling"
    display_name: "TS"
    model:
      name: "OnlineRidgeRegression"
      params: { l2_reg: 1.0 }
      one_model_per_arm: true

  - name: "UCBAlgorithm"
    display_name: "UCB (alpha=1.0)"
    params: { alpha: 1.0 }
    model:
      name: "OnlineRidgeRegression"
      params: { l2_reg: 1.0 }
      one_model_per_arm: true

metrics: [cumulative_regret, average_regret]
output:
  save_path: "./results/synthetic_test"
```

Run it:
```bash
bandit run config.yaml
```

See [docs/yaml_config.md]() for the full schema.

### 3. Streamlit GUI

```bash
bandit-gui
```

<!-- TODO: add screenshot: docs/assets/gui_main.png --><!-- ![BanditLab GUI](docs/assets/gui_main.png) -->

Features: offline policy evaluation (OPE), online benchmarking, dynamic hyperparameter tuning, interactive regret charts, exportable reports.

See [docs/gui.md]().

## Algorithms

| Category  | Algorithms |
|-----------|-----------|
| Non-contextual  | EpsilonGreedy, UCBAlgorithm, ThompsonSampling, NonContextualTSBandit, BootstrapTSBandit, FGTSAlgorithm  |
| Contextual / Linear  | LinUCBAlgorithm, CustomTSBandit, SGDTSBandit  |
| Gaussian Process  | GPTSBandit, GPUCBKernelFlexibleAlgorithm  |
| GLM  | RegCBBandit  |
| Delayed feedback  | DelayedThompsonSampling, JoulaniDelayedUCB, PatientBandits, VernadeDelayedUCB  |
| Neural  | NeuralUCBAlgorithm, NNUCBAlgorithm, NNTSBAlgorithm, NNAGPUCBAlgorithm, NNAGPUCBAdaptiveAlgorithm, NeuralBanditWithLimitedMemory_5,PFNTSAlgorithm  |

Full list with references in [docs/algorithms.md]().

## Environments

| Type  | Classes |
|-----------|-----------|
| Synthetic  | SyntheticLinearEnv, SyntheticGLMEnv, SyntheticNeuralEnv, SyntheticNonContextualEnv  |
| Dataset  | DatasetEnvironment (CSV / NPZ / JSON / folder)  |
| Logged data  | LogDataEnvironment (for offline policy evaluation)  |
| Open Bandit  | OpenBanditEnvironment (requires banditlab[obp])  |

See docs/environments.md.

## Models

See [docs/models.md]().

## CLI reference

```bash
bandit run <config.yaml>              # Run a custom experiment
bandit benchmark <name>               # Run a built-in benchmark
bandit evaluate --config <config.yaml>  # Offline policy evaluation
bandit-gui                            # Launch the Streamlit GUI
python -m mab_framework ...           # Same as `bandit`
```

Built-in benchmarks:
- **Synthetic**: linear_contextual, glm_feature, neural, noncontextual
- **Real**: movielens, mushrooms, open_bandit_all, open_bandit_men, open_bandit_ope

See [docs/cli.md]().

## Extending BanditLab

### Custom model

```python
import numpy as np

class MyModel:
    def __init__(self, feature_dim: int):
        self.w = np.zeros(feature_dim)

    def fit(self, X, y):
        # update self.w
        ...

    def sample(self, x):
        # return a sampled reward estimate
        return float(x @ self.w)
```

### Custom algorithm

```python
from mab_framework.algorithms.base import BaseAlgorithm

class MyBandit(BaseAlgorithm):
    def select_arm(self, context):
        # return arm index
        ...

    def update(self, feedbacks):
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            # update model
            ...
```

See [docs/extending.md]() and [examples/07_custom_algorithm.py]().

# BanditLab Examples

| # | File | Topic | Requires |
|---|---|---|---|
| 01 | [quickstart.py](01_quickstart.py) | Minimal TS example | core |
| 02 | [non_contextual.py](02_non_contextual.py) | ε-greedy, UCB, TS | core |
| 03 | [contextual.py](03_contextual.py) | LinUCB, TS | core |
| 04 | [neural.py](04_neural.py) | NeuralUCB | `[deep]` |
| 05 | [delayed.py](05_delayed.py) | Delayed feedback | core |
| 06 | [ope.py](06_ope.py) | Offline policy evaluation | core |
| 07 | [custom_algorithm.py](07_custom_algorithm.py) | Extend BaseAlgorithm | core |
| 08 | [custom_model.py](08_custom_model.py) | Extend BaseModel | core |
| 09 | [custom_environment.py](09_custom_environment.py) | Extend BaseEnvironment | core |
| 10 | [streamlit_launch.py](10_streamlit_launch.py) | Launch GUI | `[gui]` |

All examples are self-contained and runnable:

```bash
python examples/01_quickstart.py

## License

MIT License. See [LICENSE]().
