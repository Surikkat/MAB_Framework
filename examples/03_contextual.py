"""
03 — Contextual bandits
========================

LinUCB and Thompson Sampling with a linear reward model on a
contextual environment.

Run:
    python examples/03_contextual.py
"""

from mab_framework import (
    SyntheticLinearEnv,
    UCBAlgorithm,
    ThompsonSampling,
    OnlineRidgeRegression,
)


def run(algorithm, env, n_steps: int = 1000) -> float:
    env.reset()
    regret = 0.0
    for _ in range(n_steps):
        ctx = env.get_context()
        arm = algorithm.select_arm(ctx)
        step = env.step(arm, ctx)
        algorithm.update(step["available_rewards"])
        regret += step["optimal_reward"] - step["instant_reward"]
    return regret


def main() -> None:
    N_ARMS = 5
    CONTEXT_DIM = 10
    N_STEPS = 1000

    configs = [
        ("LinUCB (alpha=0.1)", lambda: UCBAlgorithm(
            n_arms=N_ARMS,
            model=OnlineRidgeRegression(feature_dim=CONTEXT_DIM, l2_reg=1.0),
            alpha=0.1,
        )),
        ("LinUCB (alpha=1.0)", lambda: UCBAlgorithm(
            n_arms=N_ARMS,
            model=OnlineRidgeRegression(feature_dim=CONTEXT_DIM, l2_reg=1.0),
            alpha=1.0,
        )),
        ("Thompson Sampling", lambda: ThompsonSampling(
            n_arms=N_ARMS,
            model=OnlineRidgeRegression(feature_dim=CONTEXT_DIM, l2_reg=1.0),
        )),
    ]

    print(f"Contextual benchmark — {N_ARMS} arms, {CONTEXT_DIM}-dim, {N_STEPS} steps\n")
    print(f"{'Algorithm':<25} {'Regret':>10}")
    print("-" * 37)

    for name, factory in configs:
        env = SyntheticLinearEnv(n_arms=N_ARMS, context_dim=CONTEXT_DIM)
        regret = run(factory(), env, n_steps=N_STEPS)
        print(f"{name:<25} {regret:>10.2f}")


if __name__ == "__main__":
    main()