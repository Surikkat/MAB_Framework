"""
09 — Custom environment
========================

Implement a custom environment by subclassing BaseEnvironment.

The BaseEnvironment contract for subclasses:
    - reset()              -> None
    - get_context()        -> np.ndarray
    - _step_raw(action)    -> (reward, optimal_reward)   # before delay

The base class handles:
    - delay buffering
    - building the {"available_rewards", "instant_reward", "optimal_reward"} dict
"""

import numpy as np
from mab_framework import ThompsonSampling, OnlineRidgeRegression
from mab_framework.environments.base import BaseEnvironment


class FixedRewardEnv(BaseEnvironment):
    """Each arm has a deterministic reward; the context is a dummy vector.

    Because rewards are deterministic and context is constant,
    a Thompson-Sampling bandit should converge to the best arm.
    """

    def __init__(self, rewards, context_dim: int = 1, delay_config=None):
        super().__init__(delay_config=delay_config)
        self.rewards = np.asarray(rewards, dtype=float)
        self.n_arms = len(self.rewards)
        self.context_dim = context_dim
        self._context = np.ones(context_dim, dtype=float)

    def reset(self) -> None:
        pass

    def get_context(self) -> np.ndarray:
        return self._context

    def _step_raw(self, action: int):
        reward = float(self.rewards[action])
        optimal = float(self.rewards.max())
        return reward, optimal


def main() -> None:
    N_STEPS = 1000

    env = FixedRewardEnv(rewards=[0.1, 0.5, 0.9, 0.3])
    env.reset()

    model = OnlineRidgeRegression(feature_dim=1, l2_reg=1.0)
    bandit = ThompsonSampling(n_arms=env.n_arms, model=model)

    regret = 0.0
    for _ in range(N_STEPS):
        ctx = env.get_context()
        arm = bandit.select_arm(ctx)
        step = env.step(arm, ctx)
        bandit.update(step["available_rewards"])
        regret += step["optimal_reward"] - step["instant_reward"]

    print(f"FixedRewardEnv (custom) — {N_STEPS} steps")
    print(f"  Cumulative regret: {regret:.2f}")
    print("  (Optimal arm has reward 0.9; regret is bounded by early exploration.)")


if __name__ == "__main__":
    main()