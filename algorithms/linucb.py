import numpy as np
from .base import BaseAlgorithm
from models.linear_model import OnlineRidgeRegression

class LinUCBAlgorithm(BaseAlgorithm):
    def __init__(self, n_arms: int, feature_dim: int, alpha: float = 1.0):
        super().__init__(n_arms)
        self.alpha = alpha
        self.models = [OnlineRidgeRegression(feature_dim) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        ucb_values = []
        
        for a in range(self.n_arms):
            x_a = context[a] if context.ndim > 1 else context
            
            mu, sigma = self.models[a].predict(x_a)
            ucb_values.append(mu + self.alpha * sigma)
            
        return int(np.argmax(ucb_values))

    def update(self, context: np.ndarray, action: int, reward: float):
        x_a = context[action] if context.ndim > 1 else context
        self.models[action].fit(x_a, reward)