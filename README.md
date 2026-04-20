![PyPI](https://img.shields.io/pypi/v/BanditLab)
![Python](https://img.shields.io/pypi/pyversions/BanditLab)
![License](https://img.shields.io/badge/license-MIT-green)

# BanditLab

A modular framework for experimenting with multi-armed bandits (MAB).

* 20+ algorithms (from classical to state-of-the-art)
* unified API
* plug-and-play models and datasets
* config-driven experiments

**BanditLab** is designed for both research and practical experimentation. It provides a unified interface for combining:

* bandit algorithms (UCB, Thompson Sampling, Neural, GP-based, etc.)
* predictive models (linear, GLM, neural networks, Gaussian processes)
* environments (real datasets or simulators)

---

## Installation

```bash
pip install BanditLab
```

---

## Quick Start (Python API)

```python
from mab_framework.algorithms import ThompsonSampling
from mab_framework.environments import DatasetEnvironment

env = DatasetEnvironment("data/mushroom_bandit_5000.csv")

bandit = ThompsonSampling(...)

for context in env:
    arm = bandit.select_arm(context)
    reward = env.pull(arm)
    bandit.update(context, arm, reward)
```

---

## Config-Based Experiments (Recommended)

BanditLab supports fully declarative experiment setup via configs.

Example config:

```yaml
experiment:
  name: "pool_test"
  steps: 200
  n_runs: 5

environment:
  name: "DatasetEnvironment"
  params:
    dataset_path: "data/E1_dataset.npz"

algorithms:
  - name: "ThompsonSampling"
    display_name: "Thompson Sampling (TS)"
    params: {}
    model:
      name: "OnlineRidgeRegression"
      params: { l2_reg: 1.0 }
      one_model_per_arm: true

  - name: "UCBAlgorithm"
    display_name: "LinUCB (alpha=1.0)"
    params: { alpha: 1.0 }
    model:
      name: "OnlineRidgeRegression"
      params: { l2_reg: 1.0 }
      one_model_per_arm: true

metrics:
  - cumulative_regret
  - average_regret

output:
  save_path: "./results/pool_test"
```

Run via:

```bash
python banditlab config.yaml
```

This allows running experiments without writing Python code and ensures full reproducibility.

---

## Key Features

* **20+ algorithms** — from classical (UCB, TS) to neural and GP-based methods
* **Model–Algorithm decoupling** — combine any algorithm with any reward model
* **Config-driven experiments** — easy experimentation without coding
* **Contextual bandits support**
* **Delayed feedback support** — built-in support for bandits with delays
* **Extensible** — easily implement new algorithms or models
* **Reproducible experiments** — runner, logging, and metrics included

---

## Core Design

BanditLab separates *decision-making* from *prediction*:

* **Models** learn to predict rewards from context
* **Algorithms** decide which arm to pull using model outputs

This enables flexible combinations:

* Thompson Sampling + Linear Model
* Thompson Sampling + GLM
* UCB + Neural Network
* UCB + Gaussian Process

---

## Architecture Overview

The framework is built around four components:

* **Environments** — provide contexts and rewards
* **Models** — estimate reward (typically one per arm)
* **Algorithms** — handle exploration vs exploitation
* **Runner** — executes experiment loops

---

## Example: Running a Benchmark

```bash
python scripts/run_mushrooms.py
```

This runs multiple algorithms on a real dataset and produces:

* cumulative regret plots
* average regret curves

---

## Supported Methods

### Algorithms

Includes 20+ implementations, such as:

* Epsilon-Greedy
* UCB / LinUCB
* Thompson Sampling
* Neural UCB
* GP-based methods
* GLM-based bandits

### Models

* Linear / Ridge Regression
* GLM (Laplace approximation)
* Gaussian Processes (RFF)
* Neural Networks
* LASSO-based models

---

## Project Structure

```
mab_framework/
├── algorithms/
├── models/
├── environments/
├── experiment/
└── scripts/
```

---

## Extending the Framework

### Custom Model

```python
predict(context)
update(context, reward)
```

### Custom Algorithm

```python
select_arm(context)
update(context, arm, reward)
```

All components inherit from base classes, making extension straightforward.

---

## Reproducibility

BanditLab includes:

* experiment runner
* logging utilities
* regret metrics

Designed for fair comparison of algorithms across datasets.

---

## Documentation

Detailed developer documentation is available in:

```
docs/DEVELOPMENT.md
```

---

## License

MIT License

---

## Citation

If you use BanditLab in research, please consider citing the repository.
