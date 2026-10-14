"""
02 — Non-contextual bandits
============================

Compare epsilon-greedy, UCB, and Thompson Sampling on a
non-contextual environment (constant context, stochastic rewards).

Run:
    python examples/02_non_contextual.py
"""

from mab_framework import (
    SyntheticNonContextualEnv,
    EpsilonGreedy,
    UCBAlgorithm,
    ThompsonSampling,
    OnlineRidgeRegression,
)


def run(algorithm, env, n_steps: int = 2000) -> float:
    """Run an interaction loop, return cumulative regret."""
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
    N_STEPS = 2000

    configs = [
        ("EpsilonGreedy (eps=0.1)", lambda: EpsilonGreedy(
            n_arms=N_ARMS,
            model=OnlineRidgeRegression(feature_dim=1, l2_reg=1.0),
            epsilon=0.1,
        )),
        ("UCB (alpha=1.0)", lambda: UCBAlgorithm(
            n_arms=N_ARMS,
            model=OnlineRidgeRegression(feature_dim=1, l2_reg=1.0),
            alpha=1.0,
        )),
        ("Thompson Sampling", lambda: ThompsonSampling(
            n_arms=N_ARMS,
            model=OnlineRidgeRegression(feature_dim=1, l2_reg=1.0),
        )),
    ]

    print(f"Non-contextual benchmark — {N_ARMS} arms, {N_STEPS} steps\n")
    print(f"{'Algorithm':<25} {'Regret':>10}")
    print("-" * 37)

    results = {}
    for name, factory in configs:
        env = SyntheticNonContextualEnv(n_arms=N_ARMS, context_dim=1)
        regret = run(factory(), env, n_steps=N_STEPS)
        results[name] = regret
        print(f"{name:<25} {regret:>10.2f}")

    assert all(r > 0 for r in results.values()), "All regrets must be positive"


if __name__ == "__main__":
    main()