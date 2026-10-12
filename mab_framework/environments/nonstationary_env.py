import numpy as np
from typing import Optional, Dict
from .base import BaseEnvironment


class _BaseNonStationaryEnv(BaseEnvironment):
    """
    Base class for generative non-stationary synthetic environments.
    
    Generates:
    - Static action pool X_pool in R^{K x d_x} sampled once at reset()
    - Drifting context vector theta_t via AR(1) Gauss-Markov: theta_t = 0.95 * theta_{t-1} + 0.3 * eta_t
    - Combined context for each arm: [theta_t, X_pool[a]]
    """
    def __init__(
        self,
        n_arms: int = 20,
        context_dim: int = 5,
        noise_std: float = 0.01,
        delay_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(delay_config=delay_config, **kwargs)
        self.n_arms = int(n_arms)
        self.context_dim = int(context_dim)
        self.theta_dim = int(context_dim)
        self.x_dim = int(context_dim)
        self.noise_std = float(noise_std)
        self._current_context = None
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.delay_buffer.queue.clear()
        self.delay_buffer.current_time = 0
        self.theta = np.random.randn(self.theta_dim)
        self.X_pool = np.random.randn(self.n_arms, self.x_dim)
        self._current_context = None
        self._init_weights()

    def _init_weights(self) -> None:
        raise NotImplementedError

    def get_context(self) -> np.ndarray:
        if self._current_context is None:
            theta_tile = np.tile(self.theta, (self.n_arms, 1))
            self._current_context = np.hstack([theta_tile, self.X_pool])
        return self._current_context

    def _compute_expected_rewards(self) -> np.ndarray:
        raise NotImplementedError

    def _step_raw(self, action: int) -> tuple[float, float]:
        if self._current_context is None:
            self.get_context()

        expected_rewards = self._compute_expected_rewards()
        optimal_reward = float(np.max(expected_rewards))
        reward = float(expected_rewards[action] + np.random.randn() * self.noise_std)

        # Gauss-Markov AR(1) process for context drift
        self.theta = 0.95 * self.theta + 0.3 * np.random.randn(self.theta_dim)
        self._current_context = None

        return reward, optimal_reward


class TransientRankBurstEnv(_BaseNonStationaryEnv):
    """
    Transient Rank Burst: Simulates a temporary surge in latent problem complexity.
    During normal operation, base_rank factors are active.
    Between burst_start and burst_start + burst_duration, burst_rank factors are active.
    """
    def __init__(
        self,
        n_arms: int = 20,
        context_dim: int = 5,
        noise_std: float = 0.01,
        base_rank: int = 2,
        burst_rank: int = 5,
        burst_start: int = 70,
        burst_duration: int = 50,
        delay_config: Optional[Dict] = None,
        **kwargs
    ):
        self.base_rank = max(1, int(base_rank))
        self.burst_rank = max(1, int(burst_rank))
        self.burst_start = max(0, int(burst_start))
        self.burst_duration = max(1, int(burst_duration))
        self.total_m = max(self.base_rank, self.burst_rank, int(context_dim)) + 5
        super().__init__(n_arms=n_arms, context_dim=context_dim, noise_std=noise_std, delay_config=delay_config, **kwargs)

    def _init_weights(self) -> None:
        self.W_t = np.random.randn(self.theta_dim, self.total_m) * 0.5
        self.W_x = np.random.randn(self.x_dim, self.total_m) * 0.5
        min_diag = min(self.theta_dim, self.x_dim, self.total_m)
        for i in range(min_diag):
            self.W_t[i, i] = 1.0
            self.W_x[i, i] = 1.0

    def _compute_expected_rewards(self) -> np.ndarray:
        g = self.theta @ self.W_t
        p = self.X_pool @ self.W_x
        Z = p * g

        if self.burst_start <= self.t < self.burst_start + self.burst_duration:
            active = min(self.burst_rank, self.total_m)
        else:
            active = min(self.base_rank, self.total_m)

        expected = np.zeros(self.n_arms)
        for k in range(active):
            z = Z[:, k]
            scale = 1.0 if k < 3 else 0.7
            expected += scale * np.tanh(0.8 * z) + scale * 0.3 * np.sin(4.0 * z)

        for k in range(active, self.total_m):
            expected += 0.01 * Z[:, k]

        return expected


class StepwiseRankEnv(_BaseNonStationaryEnv):
    """
    Stepwise Rank Change: Simulates a permanent changepoint shifting active latent rank.
    Before changepoint_step, initial_rank factors are active.
    After changepoint_step, final_rank factors are active.
    """
    def __init__(
        self,
        n_arms: int = 20,
        context_dim: int = 5,
        noise_std: float = 0.01,
        initial_rank: int = 6,
        final_rank: int = 2,
        changepoint_step: int = 100,
        delay_config: Optional[Dict] = None,
        **kwargs
    ):
        self.initial_rank = max(1, int(initial_rank))
        self.final_rank = max(1, int(final_rank))
        self.changepoint_step = max(0, int(changepoint_step))
        self.total_m = max(self.initial_rank, self.final_rank, int(context_dim)) + 5
        super().__init__(n_arms=n_arms, context_dim=context_dim, noise_std=noise_std, delay_config=delay_config, **kwargs)

    def _init_weights(self) -> None:
        self.W_t = np.random.randn(self.theta_dim, self.total_m) * 0.5
        self.W_x = np.random.randn(self.x_dim, self.total_m) * 0.5
        min_diag = min(self.theta_dim, self.x_dim, self.total_m)
        for i in range(min_diag):
            self.W_t[i, i] = 1.0
            self.W_x[i, i] = 1.0

    def _compute_expected_rewards(self) -> np.ndarray:
        g = self.theta @ self.W_t
        p = self.X_pool @ self.W_x
        Z = p * g

        if self.t < self.changepoint_step:
            active = min(self.initial_rank, self.total_m)
        else:
            active = min(self.final_rank, self.total_m)

        expected = np.zeros(self.n_arms)
        for k in range(active):
            z = Z[:, k]
            scale = 1.0 if k < 3 else 0.7
            expected += scale * np.tanh(0.8 * z) + scale * 0.3 * np.sin(4.0 * z)

        for k in range(active, self.total_m):
            expected += 0.01 * Z[:, k]

        return expected


class DriftingSubspaceEnv(_BaseNonStationaryEnv):
    """
    Drifting Subspace Rotation: Simulates continuous rotation of the latent subspace coordinate system.
    W(t)[:, :2] = W_0[:, :2] @ [[cos(w*t), -sin(w*t)], [sin(w*t), cos(w*t)]]
    where w = rotation_rate.
    """
    def __init__(
        self,
        n_arms: int = 20,
        context_dim: int = 5,
        noise_std: float = 0.01,
        rotation_rate: float = 0.03,
        delay_config: Optional[Dict] = None,
        **kwargs
    ):
        self.rotation_rate = max(0.0, float(rotation_rate))
        self.total_m = max(int(context_dim), 6) + 4
        super().__init__(n_arms=n_arms, context_dim=context_dim, noise_std=noise_std, delay_config=delay_config, **kwargs)

    def _init_weights(self) -> None:
        self.W_t_0 = np.random.randn(self.theta_dim, self.total_m) * 0.5
        self.W_x_0 = np.random.randn(self.x_dim, self.total_m) * 0.5
        min_diag = min(self.theta_dim, self.x_dim, self.total_m)
        for i in range(min_diag):
            self.W_t_0[i, i] = 1.0
            self.W_x_0[i, i] = 1.0

    def _compute_expected_rewards(self) -> np.ndarray:
        angle = self.rotation_rate * self.t
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])

        W_t = self.W_t_0.copy()
        W_x = self.W_x_0.copy()
        if self.total_m >= 2:
            W_t[:, :2] = self.W_t_0[:, :2] @ R
            W_x[:, :2] = self.W_x_0[:, :2] @ R

        g = self.theta @ W_t
        p = self.X_pool @ W_x
        Z = p * g

        active = min(4, self.total_m)
        expected = np.zeros(self.n_arms)
        for k in range(active):
            z = Z[:, k]
            scale = 1.0 if k < 2 else 0.7
            expected += scale * np.tanh(0.8 * z) + scale * 0.3 * np.sin(4.0 * z)

        for k in range(active, self.total_m):
            expected += 0.01 * Z[:, k]

        return expected


class HarmonicDriftEnv(_BaseNonStationaryEnv):
    """
    Harmonic Concept Drift: Simulates cyclic/seasonal variation in preferences.
    r_t = sum_k s_k [tanh(0.8 g_k p_k) + 0.25 sin(5 g_k p_k + w*t)]
    where w = drift_frequency.
    """
    def __init__(
        self,
        n_arms: int = 20,
        context_dim: int = 5,
        noise_std: float = 0.01,
        drift_frequency: float = 0.10,
        delay_config: Optional[Dict] = None,
        **kwargs
    ):
        self.drift_frequency = max(0.0, float(drift_frequency))
        self.total_m = max(int(context_dim), 6) + 4
        super().__init__(n_arms=n_arms, context_dim=context_dim, noise_std=noise_std, delay_config=delay_config, **kwargs)

    def _init_weights(self) -> None:
        self.W_t = np.random.randn(self.theta_dim, self.total_m) * 0.5
        self.W_x = np.random.randn(self.x_dim, self.total_m) * 0.5
        min_diag = min(self.theta_dim, self.x_dim, self.total_m)
        for i in range(min_diag):
            self.W_t[i, i] = 1.0
            self.W_x[i, i] = 1.0

    def _compute_expected_rewards(self) -> np.ndarray:
        g = self.theta @ self.W_t
        p = self.X_pool @ self.W_x
        Z = p * g

        active = min(4, self.total_m)
        expected = np.zeros(self.n_arms)
        for k in range(active):
            z = Z[:, k]
            s_k = 1.0 if k < 2 else 0.6
            expected += s_k * (np.tanh(0.8 * z) + 0.25 * np.sin(5.0 * z + self.drift_frequency * self.t))

        for k in range(active, self.total_m):
            expected += 0.01 * Z[:, k]

        return expected
