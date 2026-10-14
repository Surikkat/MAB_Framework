"""
01 — Quickstart
================

Minimal example: Thompson Sampling on a synthetic linear environment.

Run:
    python examples/01_quickstart.py

Expected output: a cumulative regret value in the range 10-100.
"""

from mab_framework import (
    SyntheticLinearEnv,
    ThompsonSampling,
    OnlineRidgeRegression,
)


def main() -> None:
    N_ARMS = 3
    CONTEXT_DIM = 5
    N_STEPS = 1000

    # 1. Environment
    env = SyntheticLinearEnv(n_arms=N_ARMS, context_dim=CONTEXT_DIM)
    env.reset()

    # 2. Algorithm: Thompson Sampling with a linear reward model
    model = OnlineRidgeRegression(feature_dim=CONTEXT_DIM, l2_reg=1.0)
    bandit = ThompsonSampling(n_arms=N_ARMS, model=model)

    # 3. Interaction loop
    regret = 0.0
    for _ in range(N_STEPS):
        ctx = env.get_context()
        arm = bandit.select_arm(ctx)
        step = env.step(arm, ctx)
        bandit.update(step["available_rewards"])
        regret += step["optimal_reward"] - step["instant_reward"]

    print(f"Thompson Sampling — {N_STEPS} steps")
    print(f"  Cumulative regret: {regret:.2f}")
    print(f"  Average regret:    {regret / N_STEPS:.4f}")

    # Sanity checks — these should always hold for a working library
    assert regret > 0, "Regret must be positive"
    assert regret < N_STEPS * 10, "Regret seems unreasonably large"


if __name__ == "__main__":
    main()