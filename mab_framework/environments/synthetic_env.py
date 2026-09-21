import numpy as np
from .base import BaseEnvironment


class SyntheticLinearEnv(BaseEnvironment):
    """
    Линейная среда: r = <theta_a, x> + шум.
    Классическая проверка линейной реализуемости.
    """
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
        self._current_context = np.random.randn(self.n_arms, self.context_dim)
        return self._current_context

    def _step_raw(self, action: int) -> tuple[float, float]:
        if self._current_context is None:
            self.get_context()

        expected_rewards = np.sum(self.theta * self._current_context, axis=1)
        optimal_reward = np.max(expected_rewards)
        reward = expected_rewards[action] + np.random.randn() * self.noise_std

        self._current_context = None
        return float(reward), float(optimal_reward)


class SyntheticGLMEnv(BaseEnvironment):
    """
    GLM-среда: P(r=1) = sigmoid(<theta_a, x>).
    Бинарная награда. Тестирует misspecification линейных методов.
    """
    def __init__(self, n_arms: int, context_dim: int, noise_std: float = 0.0,
                 delay_config=None):
        super().__init__(delay_config=delay_config)
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.noise_std = noise_std  # не используется, оставлен для совместимости
        self.reset()

    def reset(self) -> None:
        self.theta = np.random.randn(self.n_arms, self.context_dim)
        self.delay_buffer.queue.clear()
        self.delay_buffer.current_time = 0
        self._current_context = None

    def get_context(self) -> np.ndarray:
        self._current_context = np.random.randn(self.n_arms, self.context_dim)
        return self._current_context

    def _step_raw(self, action: int) -> tuple[float, float]:
        if self._current_context is None:
            self.get_context()

        logits = np.sum(self.theta * self._current_context, axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        optimal_reward = float(np.max(probs))
        reward = float(np.random.binomial(1, probs[action]))

        self._current_context = None
        return reward, optimal_reward


class SyntheticNeuralEnv(BaseEnvironment):
    """
    Нейронная среда: r = MLP(x) + шум.
    Двухслойный MLP с ReLU-активациями, отдельный для каждой руки.
    Создает сложную нелинейную поверхность наград.
    """
    def __init__(self, n_arms: int, context_dim: int, hidden_dim: int = 32,
                 noise_std: float = 0.1, delay_config=None):
        super().__init__(delay_config=delay_config)
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.reset()

    def reset(self) -> None:
        # Случайные веса MLP для каждой руки
        self.W1 = np.random.randn(self.n_arms, self.context_dim, self.hidden_dim) * 0.5
        self.b1 = np.random.randn(self.n_arms, self.hidden_dim) * 0.1
        self.W2 = np.random.randn(self.n_arms, self.hidden_dim) * 0.5
        self.b2 = np.random.randn(self.n_arms) * 0.1

        self.delay_buffer.queue.clear()
        self.delay_buffer.current_time = 0
        self._current_context = None

    def get_context(self) -> np.ndarray:
        self._current_context = np.random.randn(self.n_arms, self.context_dim)
        return self._current_context

    def _step_raw(self, action: int) -> tuple[float, float]:
        if self._current_context is None:
            self.get_context()

        # MLP: h = relu(x @ W1 + b1); y = h @ W2 + b2
        h = np.maximum(
            0.0,
            np.einsum('ad,adh->ah', self._current_context, self.W1) + self.b1
        )  # (n_arms, hidden_dim)
        y = np.einsum('ah,ah->a', h, self.W2) + self.b2  # (n_arms,)

        optimal_reward = float(np.max(y))
        reward = float(y[action] + np.random.randn() * self.noise_std)

        self._current_context = None
        return reward, optimal_reward


class SyntheticNonContextualEnv(BaseEnvironment):
    """
    Безконтекстная среда: у каждой руки фиксированное среднее mu_a.
    Награда = mu_a + шум. Контекст — нулевой (не несет информации).
    Служит baseline для non-contextual алгоритмов.
    """
    def __init__(self, n_arms: int, context_dim: int = 1,
                 noise_std: float = 1.0, delay_config=None):
        super().__init__(delay_config=delay_config)
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.noise_std = noise_std
        self.reset()

    def reset(self) -> None:
        self.means = np.random.randn(self.n_arms)
        self.delay_buffer.queue.clear()
        self.delay_buffer.current_time = 0
        self._current_context = None

    def get_context(self) -> np.ndarray:
        self._current_context = np.zeros((self.n_arms, self.context_dim))
        return self._current_context

    def _step_raw(self, action: int) -> tuple[float, float]:
        optimal_reward = float(np.max(self.means))
        reward = float(self.means[action] + np.random.randn() * self.noise_std)
        self._current_context = None
        return reward, optimal_reward
