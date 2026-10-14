"""
06 — Offline Policy Evaluation
================================

Evaluate a candidate policy on logged data (no online interaction).
Uses IPS / SNIPS / DM / DR estimators.

Run:
    python examples/06_ope.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from mab_framework import (
    LogDataEnvironment,
    ThompsonSampling,
    OnlineRidgeRegression,
)
from mab_framework.experiment.ope_pipeline import OPEPipeline


def make_synthetic_logs(path: Path, n: int = 5000, n_arms: int = 3, context_dim: int = 5) -> None:
    """Generate a synthetic logged dataset with propensity scores."""
    rng = np.random.default_rng(0)

    contexts = rng.normal(size=(n, context_dim))
    # Random logging policy
    actions = rng.integers(0, n_arms, size=n)
    pscore = np.full(n, 1.0 / n_arms)  # uniform propensity

    # True reward: linear model + arm bias
    true_w = rng.normal(size=(n_arms, context_dim))
    arm_bias = np.array([0.0, 0.5, 1.0])[:n_arms]
    rewards = np.einsum("ij,ij->i", contexts, true_w[actions]) + arm_bias[actions]
    rewards += rng.normal(scale=0.1, size=n)

    df = pd.DataFrame(contexts, columns=[f"ctx_{i}" for i in range(context_dim)])
    df["action"] = actions
    df["reward"] = rewards
    df["pscore"] = pscore
    df.to_csv(path, index=False)


def main() -> None:
    N_ARMS = 3
    CONTEXT_DIM = 5
    LOG_PATH = Path("/tmp/banditlab_example_logs.csv")

    if not LOG_PATH.exists():
        print(f"Generating synthetic logs at {LOG_PATH} ...")
        make_synthetic_logs(LOG_PATH, n=5000, n_arms=N_ARMS, context_dim=CONTEXT_DIM)

    env = LogDataEnvironment(
        log_path=str(LOG_PATH),
        context_columns="auto",
        action_column="action",
        reward_column="reward",
        pscore_column="pscore",
        n_actions=N_ARMS,
    )

    def algorithm_factory():
        model = OnlineRidgeRegression(feature_dim=CONTEXT_DIM, l2_reg=1.0)
        return ThompsonSampling(n_arms=N_ARMS, model=model)

    pipeline = OPEPipeline(
        env=env,
        algorithm_factory=algorithm_factory,
        candidate_name="ThompsonSampling",
        estimator_names=["ips", "snips"],
        train_ratio=0.7,
        n_runs=3,
        seed=42,
    )

    result = pipeline.run()
    print("OPE results:")
    print(result)


if __name__ == "__main__":
    main()