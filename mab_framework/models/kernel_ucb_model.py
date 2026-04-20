"""KernelUCB model — exact reproduction of FGTS_LASSO/kernelUCB.py.

Stores per-arm history and uses RBF kernel for UCB.
"""
import numpy as np
from .base import BaseModel


class KernelUCBModel(BaseModel):
    def __init__(self, kernel='rbf', gamma: float = 1.0, lam: float = 1e-3, beta: float = 1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.X_hist = []
        self.Y_hist = []

    def _rbf_kernel(self, X1, X2):
        sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
        return np.exp(-self.gamma * sq_dist)

    def fit(self, x: np.ndarray, y: float):
        self.X_hist.append(x)
        self.Y_hist.append(y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_row = x.reshape(1, -1)
        if len(self.X_hist) == 0:
            return np.array([np.inf]), np.array([np.inf])
        X_a = np.vstack(self.X_hist)
        Y_a = np.array(self.Y_hist)

        K_mat = self._rbf_kernel(X_a, X_a) + self.lam * np.eye(len(X_a))
        k_vec = self._rbf_kernel(X_a, x_row).flatten()

        alpha = np.linalg.solve(K_mat, Y_a)
        mu = k_vec @ alpha

        sigma = np.sqrt(self.beta * (1 - k_vec @ np.linalg.solve(K_mat, k_vec)))

        return np.array([mu]), np.array([sigma])

    def sample(self, x: np.ndarray) -> np.ndarray:
        mu, sigma = self.predict(x)
        return mu + np.random.randn() * sigma
