import numpy as np
from .base import BaseModel

class BootstrapEnsembleModel(BaseModel):
    def __init__(self, feature_dim: int, n_models: int = 10, lr: float = 0.01, fixed_std: float = 0.1, bootstrap_prob: float = 0.8):
        self.d = feature_dim
        self.n_models = n_models
        self.lr = lr
        self.fixed_std = fixed_std
        self.bootstrap_prob = bootstrap_prob
        self.reset()

    def reset(self):
        self.models = [np.zeros(self.d) for _ in range(self.n_models)]

    def fit(self, x: np.ndarray, y: float) -> None:
        for i in range(self.n_models):
            if np.random.rand() < self.bootstrap_prob:
                pred = self.models[i] @ x
                error = pred - y
                grad = error * x
                self.models[i] -= self.lr * grad

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = np.array([m @ x for m in self.models])
        return np.array([np.mean(preds)]), np.array([np.std(preds) + self.fixed_std])

    def sample(self, x: np.ndarray) -> np.ndarray:
        model_idx = np.random.randint(self.n_models)
        if x.ndim > 1:
            return x.dot(self.models[model_idx])
        return np.array([self.models[model_idx] @ x])
