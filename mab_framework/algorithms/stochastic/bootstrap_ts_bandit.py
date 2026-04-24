import numpy as np
from typing import List, Dict, Any
from ..base import BaseAlgorithm

class BootstrapTSBandit(BaseAlgorithm):
    def __init__(self, n_arms, d, n_models=10, lr=0.01, fixed_std=0.1, bootstrap_prob=0.8, model=None):
        super().__init__(n_arms, model)
        self.d = d
        self.n_models = n_models
        self.lr = lr
        self.fixed_std = fixed_std
        self.bootstrap_prob = bootstrap_prob
        self.models = [
            [np.zeros(d) for _ in range(n_arms)] for _ in range(n_models)
        ]
        self.data = []

    def select_arm(self, context):
        """
        context: shape (n_arms, d)
        """
        model_idx = np.random.randint(self.n_models)
        weights = self.models[model_idx]
        means = np.array([weights[i] @ context[i] for i in range(self.n_arms)])
        return int(np.argmax(means))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            x_a = context[action] if context.ndim > 1 else context

            self.data.append((x_a, action, reward))
            for i in range(self.n_models):
                if np.random.rand() < self.bootstrap_prob:
                    weights = self.models[i][action]
                    pred = weights @ x_a
                    error = pred - reward
                    grad = error * x_a
                    self.models[i][action] -= self.lr * grad

