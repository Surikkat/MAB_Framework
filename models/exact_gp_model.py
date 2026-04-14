"""ExactGPModel — exact reproduction of FGTS_LASSO/gp_ts.py.

Full GP with RBF kernel (O(n³) inverse per step). Per-arm history.
"""
import numpy as np
from .base import BaseModel


class ExactGPModel(BaseModel):
    def __init__(self, gamma: float = 1.0, sigma_noise: float = 0.1, window_size: int = None):
        self.gamma = gamma
        self.sigma_noise = sigma_noise
        self.window_size = window_size
        self.X_hist = []
        self.Y_hist = []

    def _rbf_kernel(self, X1, X2):
        sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
        return np.exp(-self.gamma * sq_dist)

    def fit(self, x: np.ndarray, y: float):
        self.X_hist.append(x)
        self.Y_hist.append(y)
        if self.window_size is not None and len(self.X_hist) > self.window_size:
            self.X_hist.pop(0)
            self.Y_hist.pop(0)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_row = x.reshape(1, -1)
        if len(self.X_hist) == 0:
            return np.array([0.0]), np.array([1.0])
        X_a = np.vstack(self.X_hist)
        Y_a = np.array(self.Y_hist)

        K_mat = self._rbf_kernel(X_a, X_a) + self.sigma_noise**2 * np.eye(len(X_a))
        k_star = self._rbf_kernel(X_a, x_row).flatten()

        K_inv = np.linalg.inv(K_mat)
        mu = k_star @ K_inv @ Y_a
        cov = 1 - k_star @ K_inv @ k_star

        return np.array([mu]), np.array([cov])

    def sample(self, x: np.ndarray) -> np.ndarray:
        mu, cov = self.predict(x)
        if len(self.X_hist) == 0:
            return np.array([np.inf])
        f_sample = np.random.normal(mu, np.sqrt(cov))
        return np.array([f_sample])
