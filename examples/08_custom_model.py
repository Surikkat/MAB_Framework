"""
08 — Custom model
==================

Implement a custom reward model by subclassing BaseModel.

The BaseModel contract:
    - fit(x, y)          -> None
    - predict(x)         -> (mean, uncertainty), both np.ndarray
    - sample(x)          -> float (used by Thompson Sampling)

Here: a running-mean model that ignores the context entirely.
"""

import numpy as np
from mab_framework import SyntheticNonContextualEnv, ThompsonSampling
from mab_framework.models.base import BaseModel


class RunningMeanModel(BaseModel):
    """Estimates expected reward as the running mean of observed rewards."""

    def __init__(self, prior_mean: float = 0.0, prior_std: float = 1.0):
        self.mean = float(prior_mean)
        self.n = 0
        self.prior_std = float(prior_std)

    def fit(self, x, y) -> None:
        self.n += 1
        self.mean += (float(y) - self.mean) / self.n

    def predict(self, x):
        # Posterior uncertainty shrinks as 1/sqrt(n)
        uncertainty = self.prior_std / (self.n ** 0.5 + 1.0)
        return np.array([self.mean]), np.array([uncertainty])

    def sample(self, x) -> float:
        # Thompson-style sampling with shrinking std
        std = self.prior_std / (self.n ** 0.5 + 1.0)
        return float(np.random.normal(self.mean, std))


def main() -> None:
    N_ARMS = 3
    N_STEPS = 1000

    env = SyntheticNonContextualEnv(n_arms=N_ARMS, context_dim=1)
    env.reset()

    model = RunningMeanModel()
    bandit = ThompsonSampling(n_arms=N_ARMS, model=model)

    regret = 0.0
    for _ in range(N_STEPS):
        ctx = env.get_context()
        arm = bandit.select_arm(ctx)
        step = env.step(arm, ctx)
        bandit.update(step["available_rewards"])
        regret += step["optimal_reward"] - step["instant_reward"]

    print(f"RunningMeanModel (custom) — {N_STEPS} steps")
    print(f"  Cumulative regret: {regret:.2f}")


if __name__ == "__main__":
    main()