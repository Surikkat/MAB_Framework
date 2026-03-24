import numpy as np
from .base import BaseAlgorithm
from models.neural_network import NeuralLinearModel

class NeuralUCBAlgorithm(BaseAlgorithm):
    def __init__(self, n_arms: int, feature_dim: int, hidden_dim: int = 32, alpha: float = 1.0, lr: float = 0.01):
        super().__init__(n_arms)
        self.alpha = alpha
        self.models =[NeuralLinearModel(feature_dim, hidden_dim, lr) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        ucb_values =[]
        for a in range(self.n_arms):
            x_a = context[a] if context.ndim > 1 else context
            mu, sigma = self.models[a].predict(x_a)
            ucb_values.append(mu + self.alpha * sigma)
            
        return int(np.argmax(ucb_values))

    def update(self, context: np.ndarray, action: int, reward: float):
        x_a = context[action] if context.ndim > 1 else context
        self.models[action].fit(x_a, reward)