import numpy as np
from .base import BaseAlgorithm
from models.fgts_model import FGTSModel

class FGTSAlgorithm(BaseAlgorithm):
    def __init__(self, n_arms: int, feature_dim: int, lasso_alpha=None, lasso_start=100, lasso_period=100, window=500):
        super().__init__(n_arms)
        self.models =[
            FGTSModel(feature_dim, lasso_alpha, lasso_start, lasso_period, window) 
            for _ in range(n_arms)
        ]

    def select_arm(self, context: np.ndarray) -> int:
        expected_rewards = []
        for a in range(self.n_arms):
            x_a = context[a] if context.ndim > 1 else context
            mu, _ = self.models[a].predict(x_a)
            expected_rewards.append(mu)
            
        return int(np.argmax(expected_rewards))

    def update(self, context: np.ndarray, action: int, reward: float):
        x_a = context[action] if context.ndim > 1 else context
        self.models[action].fit(x_a, reward)