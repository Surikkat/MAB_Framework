import numpy as np
from .base import BaseAlgorithm

class EpsilonGreedy(BaseAlgorithm):

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self, context: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        
        return int(np.argmax(self.values))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = value + (reward - value) / n