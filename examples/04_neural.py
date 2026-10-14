"""
04 — Neural bandits
====================

NeuralUCB on a synthetic neural environment.

Requires: pip install banditlab[deep]

Run:
    python examples/04_neural.py
"""

from mab_framework import (
    SyntheticNeuralEnv,
    NeuralUCBAlgorithm,
    NeuralUCBModel,
)


def main() -> None:
    N_ARMS = 3
    CONTEXT_DIM = 5
    N_STEPS = 300

    env = SyntheticNeuralEnv(
        n_arms=N_ARMS,
        context_dim=CONTEXT_DIM,
        hidden_dim=32,
    )
    env.reset()

    model = NeuralUCBModel(
        feature_dim=CONTEXT_DIM,
        m=64,
        L=2,
    )
    bandit = NeuralUCBAlgorithm(
        n_arms=N_ARMS,
        model=model,
        T=N_STEPS,
        lambd=1.0,
        nu=1.0,
        delta=0.01,
    )

    regret = 0.0
    for _ in range(N_STEPS):
        ctx = env.get_context()
        arm = bandit.select_arm(ctx)
        step = env.step(arm, ctx)
        bandit.update(step["available_rewards"])
        regret += step["optimal_reward"] - step["instant_reward"]

    print(f"NeuralUCB — {N_STEPS} steps")
    print(f"  Cumulative regret: {regret:.2f}")


if __name__ == "__main__":
    main()