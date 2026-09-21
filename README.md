![PyPI](https://img.shields.io/pypi/v/BanditLab)
![Python](https://img.shields.io/pypi/pyversions/BanditLab)
![License](https://img.shields.io/badge/license-MIT-green)

# BanditLab

A modular framework for reproducible contextual bandit research with three levels of accessibility.

- 20+ algorithms (from classical to state-of-the-art)
- unified API
- plug-and-play models and datasets
- config-driven experiments
- **Streamlit GUI** — no coding required

**BanditLab** is designed for both research and practical experimentation. It provides three complementary levels of interaction:

1. **Python Library** — maximum flexibility for researchers and engineers
2. **YAML Configuration** — reproducible experiments without coding
3. **Streamlit Web Interface** — accessible GUI for non-programmers

---

## Installation

```bash
pip install BanditLab
```
---

## Three Levels of Accessibility

### Level 1: Python Library (Maximum Flexibility)

Full programmatic control over every component:

'''python
from mab_framework.algorithms import ThompsonSampling
from mab_framework.environments import DatasetEnvironment
from mab_framework.models import OnlineRidgeRegression

env = DatasetEnvironment("data/mushroom_bandit_5000.csv")
model = OnlineRidgeRegression(l2_reg=1.0)
bandit = ThompsonSampling(model=model, n_arms=env.n_arms)

for context in env:
    arm = bandit.select_arm(context)
    reward = env.pull(arm)
    bandit.update(context, arm, reward)
'''

Target audience: Academic researchers, data scientists, ML engineers.

### Level 2: YAML Configuration (Reproducible Experiments)

Declarative experiment specification without writing code:

'''yaml
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
'''

Run with a single command:
'''bash
python -m banditlab config.yaml
'''

Target audience: Academic researchers, data scientists, ML engineers.

### Level 3: Streamlit Web Interface (Maximum Accessibility)

Intuitive web-based GUI — no programming required:

'''bash
streamlit run streamlit_bandit/app.py
'''

Features:

  - Offline Policy Evaluation (OPE) — evaluate algorithms on historical logs using DM, IPS, and DR estimators

  - Online Benchmarking — run algorithms on synthetic and real-world environments with regret visualization

  - Dynamic hyperparameter configuration

  - Real-time progress tracking

  - Interactive charts and leaderboards

  - Exportable reports

Target audience: Business analysts, product managers, educators, domain experts.

## Key Features

- 20+ algorithms — from classical (UCB, TS) to neural and GP-based methods

- Model–Algorithm decoupling — combine any algorithm with any reward model

- Three levels of use — Python API, YAML configs, or Streamlit GUI

- Contextual bandits support

- Delayed feedback support — built-in support for bandits with delays

- Offline Policy Evaluation — DM, IPS, DR estimators

- Extensible — easily implement new algorithms or models

- Reproducible experiments — runner, logging, and metrics included

## Core Design

BanditLab separates decision-making from prediction:

  - Models learn to predict rewards from context

  - Algorithms decide which arm to pull using model outputs

This enables flexible combinations:

  - Thompson Sampling + Linear Model

  - Thompson Sampling + GLM

  - UCB + Neural Network

  - UCB + Gaussian Process

## Supported Methods

### Algorithms (20+)


Models

    OnlineRidgeRegression (linear)

    GLMLaplaceModel (generalized linear)

    GPRFFModel, ExactGPModel (Gaussian processes)

    NeuralUCBModel, NNUCBModel, NNAGPModel (neural networks)

    FGTSModel, FGTSLassoModel

    BootstrapEnsembleModel

    SGDModel

    KernelUCBModel

Architecture

The framework is built around four components:

    Environments — provide contexts and rewards (synthetic or data-driven)

    Models — estimate reward (typically one per arm)

    Algorithms — handle exploration vs exploitation

    Runner — executes experiment loops

Project Structure
text

mab_framework/           # Core library
├── algorithms/          # Bandit algorithms
├── models/              # Reward prediction models
├── environments/        # Data sources and simulators
├── experiment/          # Runner and logging
└── scripts/             # Utility scripts

streamlit_bandit/        # Streamlit GUI application
├── app.py               # Main entry point
├── core/                # Core logic (candidates, hyperparams, online experiments)
├── data/                # Demo datasets
└── utils/               # Visualization and export utilities

Extending the Framework
Custom Model
python

class CustomModel:
    def fit(self, X, y): ...
    def predict(self, X): ...
    def get_uncertainty(self, X): ...

Custom Algorithm
python

class CustomAlgorithm:
    def select_arm(self, context): ...
    def update(self, context, arm, reward): ...

All components inherit from base classes, making extension straightforward.

License

MIT License
Citation

If you use BanditLab in research, please consider citing the repository:
bibtex

@misc{banditlab2024,
    author = {BanditLab Contributors},
    title = {BanditLab: A Modular Framework for Contextual Bandit Research},
    year = {2024},
    publisher = {GitHub},
    url = {https://github.com/Surikkat/MAB_Framework}
}

Links

    PyPI: https://pypi.org/project/BanditLab/

    GitHub: https://github.com/Surikkat/MAB_Framework

    Documentation: Coming soon
