"""
07 — Custom algorithm
======================

Implement a custom bandit algorithm by subclassing BaseAlgorithm.

Here: a pure greedy bandit that always picks the arm with the
highest predicted reward. Because it never explores, its regret
is high — this illustrates why exploration matters.
"""

import numpy as np
from mab_framework import SyntheticLinearEnv, OnlineRidgeRegression
from mab_framework.algorithms.base import BaseAlgorithm


class GreedyBandit(BaseAlgorithm):
    """Always pick the arm with the highest predicted reward."""

    def select_arm(self, context: np.ndarray) -> int:
        predictions = []
        for a in range(self.n_arms):
            # In a contextual env, context has shape (n_arms, context_dim);
            # row `a` is the context for arm `a`.
            x_a = context[a] if context.ndim > 1 else context
            mean, _uncertainty = self.model[a].predict(x_a)
            # predict returns (mean, uncertainty); both are np.ndarray
            predictions.append(float(np.asarray(mean).ravel()[0]))
        return int(np.argmax(predictions))

    def update(self, feedbacks):
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            x_a = context[action] if context.ndim > 1 else context
            self.model[action].fit(x_a, reward)


def main() -> None:
    N_ARMS = 3
    CONTEXT_DIM = 5
    N_STEPS = 500

    env = SyntheticLinearEnv(n_arms=N_ARMS, context_dim=CONTEXT_DIM)
    env.reset()

    # One model per arm (BaseAlgorithm deep-copies it internally)
    model = OnlineRidgeRegression(feature_dim=CONTEXT_DIM, l2_reg=1.0)
    bandit = GreedyBandit(n_arms=N_ARMS, model=model)

    regret = 0.0
    for _ in range(N_STEPS):
        ctx = env.get_context()
        arm = bandit.select_arm(ctx)
        step = env.step(arm, ctx)
        bandit.update(step["available_rewards"])
        regret += step["optimal_reward"] - step["instant_reward"]

    print(f"GreedyBandit (custom) — {N_STEPS} steps")
    print(f"  Cumulative regret: {regret:.2f}")
    print("  (Pure greedy → high regret is expected: no exploration.)")


if __name__ == "__main__":
    main()