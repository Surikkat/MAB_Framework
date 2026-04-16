import numpy as np
from typing import List, Dict, Any
from .base import BaseAlgorithm


class CustomTSBandit(BaseAlgorithm):
    def __init__(self, model, n_arms, context_dim, dist_type="normal", use_hashing=False, hash_dim=None, random_state=42):
        super().__init__(n_arms, model)
        self.dist_type = dist_type
        self.use_hashing = use_hashing
        self.context_dim = context_dim

        if use_hashing:
            assert hash_dim is not None, "You must specify hash_dim when use_hashing=True"
            self.hash_dim = hash_dim
            rng = np.random.RandomState(random_state)
            # случайная матрица проекции: context_dim -> hash_dim
            self.proj_matrix = rng.choice([-1, 1], size=(self.context_dim, self.hash_dim)).astype(np.float32)
        else:
            self.hash_dim = context_dim
            self.proj_matrix = None

    def _transform(self, context: np.ndarray) -> np.ndarray:
        if not self.use_hashing:
            return context

        if context.ndim == 1:
            # одиночный вектор
            return context @ self.proj_matrix
        else:
            # батч: (n_arms, d) @ (d, hash_dim) -> (n_arms, hash_dim)
            return context @ self.proj_matrix

    def select_arm(self, context: np.ndarray) -> int:
        transformed_context = self._transform(context)
        params = self.model.predict(transformed_context)

        rewards = []
        for i in range(self.n_arms):
            p1, p2 = params[i]
            if self.dist_type == "normal":
                sampled = np.random.normal(loc=p1, scale=p2)
            elif self.dist_type == "beta":
                sampled = np.random.beta(a=max(p1, 1e-2), b=p2)
            else:
                raise ValueError("Unsupported dist_type")
            rewards.append(sampled)

        return int(np.argmax(rewards))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            context_transformed = self._transform(context)
            self.model.partial_fit(context_transformed, action, reward)

    def train(self):
        pass
