import numpy as np
from typing import List, Dict, Any
from ..base import BaseAlgorithm
from ...models.base import BaseModel

class BootstrapEnsembleModel(BaseModel):
    """
    Ensemble model for Bootstrap Thompson Sampling.
    
    References
    ----------
    Osband, I., Blundell, C., Pritzel, A., & Van Roy, B. (2016). 
    "Deep Exploration via Bootstrapped DQN." NeurIPS.
    """
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


class BootstrapTSBandit(BaseAlgorithm):
    """
    Bootstrap Thompson Sampling Algorithm.
    
    Uses an ensemble of bootstrapped models to approximate the posterior 
    distribution for exploration.
    
    References
    ----------
    Osband, I., Blundell, C., Pritzel, A., & Van Roy, B. (2016). 
    "Deep Exploration via Bootstrapped DQN." NeurIPS.
    """
    def __init__(self, n_arms, d, model=None):
        super().__init__(n_arms, model)
        self.d = d
        if self.model is None or not isinstance(self.model, list):
            raise ValueError("model must be a list of BootstrapEnsembleModel (one per arm)")

    def select_arm(self, context):
        """
        context: shape (n_arms, d)
        """
        means = np.array([self.model[i].sample(context[i])[0] for i in range(self.n_arms)])
        return int(np.argmax(means))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            x_a = context[action] if context.ndim > 1 else context
            self.model[action].fit(x_a, reward)

