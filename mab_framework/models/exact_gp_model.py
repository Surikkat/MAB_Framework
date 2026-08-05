import numpy as np
from collections import deque
from .base import BaseModel


class ExactGPModel(BaseModel):
    """
    Exact Gaussian Process Model with RBF kernel and per-arm history.
    """
    def __init__(self, gamma: float = 1.0, sigma_noise: float = 0.1, window_size: int = None, **kwargs):
        self.gamma = gamma
        self.sigma_noise = sigma_noise
        self.window_size = window_size
        self.X_hist = deque(maxlen=window_size) if window_size else []
        self.Y_hist = deque(maxlen=window_size) if window_size else []

    def _rbf_kernel(self, X1, X2):
        sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
        return np.exp(-self.gamma * sq_dist)

    def fit(self, x: np.ndarray, y: float):
        self.X_hist.append(x)
        self.Y_hist.append(y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_row = x.reshape(1, -1)
        if len(self.X_hist) == 0:
            return np.array([0.0]), np.array([1.0])
        X_a = np.vstack(self.X_hist)
        Y_a = np.array(self.Y_hist)

        K_mat = self._rbf_kernel(X_a, X_a) + self.sigma_noise**2 * np.eye(len(X_a))
        k_star = self._rbf_kernel(X_a, x_row).flatten()

        L = np.linalg.cholesky(K_mat)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_a))
        mu = k_star @ alpha
        v = np.linalg.solve(L, k_star)
        cov = 1 - v @ v

        return np.array([mu]), np.array([cov])

    def sample(self, x: np.ndarray) -> np.ndarray:
        mu, cov = self.predict(x)
        if len(self.X_hist) == 0:
            return np.array([np.inf])
        f_sample = np.random.normal(mu, np.sqrt(cov))
        return np.array([f_sample])
