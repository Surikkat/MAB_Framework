import numpy as np
from typing import List, Dict, Any
from ..base import BaseAlgorithm


class CustomTSBandit(BaseAlgorithm):
    """
    Custom Thompson Sampling Bandit with reward-space sampling.
    
    References
    ----------
    Suraveikin, E., Omirzak, D., Sultimov, R., & Maximov, Y. (2026). 
    "Efficient Contextual Bandit Learning via Reward-Space Sampling 
    and Online Optimization." AAAI.
    """
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
        if isinstance(params, tuple) and len(params) == 2 and hasattr(params[0], '__len__'):
            params = list(zip(params[0], params[1]))

        rewards = []
        for i in range(self.n_arms):
            p1, p2 = params[i]
            if hasattr(p1, '__len__'):
                p1 = float(p1[0])
            if hasattr(p2, '__len__'):
                p2 = float(p2[0])
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
            ctx = context[action] if context.ndim > 1 else context
            context_transformed = self._transform(ctx)
            if hasattr(self.model, 'partial_fit'):
                try:
                    self.model.partial_fit(context_transformed, action, reward)
                except TypeError:
                    self.model.partial_fit(context_transformed, reward)
            elif hasattr(self.model, 'fit'):
                try:
                    self.model.fit(context_transformed, action, reward)
                except TypeError:
                    self.model.fit(context_transformed, reward)
            elif hasattr(self.model, 'update'):
                self.model.update(context_transformed, action, reward)

    def train(self):
        pass
