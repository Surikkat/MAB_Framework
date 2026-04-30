import numpy as np
from .base import BaseEnvironment

class SyntheticLinearEnv(BaseEnvironment):
    def __init__(self, n_arms: int, context_dim: int, noise_std: float = 0.1, delay_config=None):
        super().__init__(delay_config=delay_config)
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.noise_std = noise_std
        self.reset()

    def reset(self) -> None:
        self.theta = np.random.randn(self.n_arms, self.context_dim)
        self.delay_buffer.queue.clear()
        self.delay_buffer.current_time = 0
        self._current_context = None

    def get_context(self) -> np.ndarray:
        # Generate new context if it hasn't been generated for this round
        self._current_context = np.random.randn(self.n_arms, self.context_dim)
        return self._current_context

    def _step_raw(self, action: int) -> tuple[float, float]:
        if self._current_context is None:
            self.get_context()
        
        expected_rewards = np.sum(self.theta * self._current_context, axis=1)
        optimal_reward = np.max(expected_rewards)
        
        expected_reward_action = expected_rewards[action]
        reward = expected_reward_action + np.random.randn() * self.noise_std
        
        self._current_context = None # Reset context for the next step
        return reward, optimal_reward
