"""
05 — Delayed feedback
======================

Bandits where the reward arrives several steps after the action
(e.g., a user converts hours after seeing an ad).

This example compares DelayedThompsonSampling across different
delay values, showing that larger delays slow down learning.

Run:
    python examples/05_delayed.py
"""

from mab_framework import (
    SyntheticLinearEnv,
    DelayedThompsonSampling,
    OnlineRidgeRegression,
)


def run_experiment(delay: int, n_arms: int = 3, context_dim: int = 5,
                   n_steps: int = 1000) -> float:
    """Run one delayed-feedback experiment, return cumulative regret."""
    env = SyntheticLinearEnv(
        n_arms=n_arms,
        context_dim=context_dim,
        delay_config={"type": "fixed", "value": delay},
    )
    env.reset()

    model = OnlineRidgeRegression(feature_dim=context_dim, l2_reg=1.0)
    bandit = DelayedThompsonSampling(n_arms=n_arms, model=model)

    regret = 0.0
    for _ in range(n_steps):
        ctx = env.get_context()
        arm = bandit.select_arm(ctx)
        step = env.step(arm, ctx)
        bandit.update(step["available_rewards"])
        regret += step["optimal_reward"] - step["instant_reward"]
    return regret


def main() -> None:
    N_STEPS = 1000
    delays = [0, 5, 20]

    print(f"DelayedThompsonSampling — {N_STEPS} steps per run\n")
    print(f"{'Delay':>6}  {'Regret':>10}")
    print("-" * 20)

    for delay in delays:
        regret = run_experiment(delay, n_steps=N_STEPS)
        print(f"{delay:>6}  {regret:>10.2f}")


if __name__ == "__main__":
    main()