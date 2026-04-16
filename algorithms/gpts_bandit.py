"""GPTSBandit — exact reproduction of cMAB_bandits/bandits/GPTSBandit.py.

Single-GP Thompson Sampling with RFF. Includes the original broadcasting bug
where `np.outer(phi, phi)` is calculated with the full `context` instead of `context[arm]`.
"""
import numpy as np
from typing import List, Dict, Any
from .base import BaseAlgorithm


class GPTSBandit(BaseAlgorithm):
    def __init__(self,
                 n_arms: int,
                 d: int,
                 kernel_scale: float = 1.0,
                 lengthscale: float = 1.0,
                 n_features: int = 500,
                 lambda_prior: float = 1.0,
                 nu0: float = 1.0,
                 model=None):
        super().__init__(n_arms, model)
        self.d = d
        self.scale = kernel_scale
        self.ls = lengthscale
        self.D = n_features
        self.lambda0 = lambda_prior
        self.nu0 = nu0

        self.W = np.random.normal(0, (2 * np.pi) / self.ls, size=(d, self.D))
        self.b = np.random.uniform(0, 2 * np.pi, size=self.D)

        self.A = self.lambda0 * np.eye(self.D)
        self.v = np.zeros(self.D)
        self.t = 0

    def _phi(self, x: np.ndarray) -> np.ndarray:
        proj = x.dot(self.W) + self.b
        return np.sqrt(2 * self.scale / self.D) * np.cos(proj)

    def select_arm(self, context: np.ndarray) -> int:
        phi_all = self._phi(context)
        try:
            L = np.linalg.cholesky(self.A)
            mu = np.linalg.solve(L.T, np.linalg.solve(L, self.v))
        except np.linalg.LinAlgError:
            mu = np.linalg.solve(self.A, self.v)

        nu_t = self.nu0 * np.sqrt(np.log(self.t + 2))

        try:
            eps = np.random.randn(self.D)
            theta = mu + nu_t * np.linalg.solve(L.T, eps)
        except Exception:
            cov = (nu_t ** 2) * np.linalg.inv(self.A)
            theta = np.random.multivariate_normal(mu, cov)

        scores = phi_all @ theta
        return int(np.argmax(scores))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            x_a = context[action] if context.ndim > 1 else context
            phi = self._phi(x_a)
            self.A += np.outer(phi, phi)
            self.v += phi * reward
            self.t += 1

