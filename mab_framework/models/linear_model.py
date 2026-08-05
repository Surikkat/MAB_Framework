import numpy as np
from .base import BaseModel

class OnlineRidgeRegression(BaseModel):
    """
    Online Ridge Regression model for Linear Contextual Bandits (LinUCB/LinTS).
    Uses Sherman-Morrison rank-1 update for O(d^2) updates and fast prediction.
    """
    def __init__(self, feature_dim: int, l2_reg: float = 1.0, nu: float = 1.0):
        self.feature_dim = feature_dim
        self.nu = nu
        self.l2_reg = l2_reg
        self.b = np.zeros(feature_dim)
        self.A_inv = np.eye(feature_dim) / l2_reg
        self.t = 0

    def fit(self, x: np.ndarray, y: float):
        x_vec = x.flatten()
        # Sherman-Morrison rank-1 update:
        # A_inv_new = A_inv - (A_inv @ x @ x^T @ A_inv) / (1 + x^T @ A_inv @ x)
        v = self.A_inv @ x_vec
        denom = 1.0 + np.dot(x_vec, v)
        self.A_inv -= np.outer(v, v) / denom

        self.b += y * x_vec
        self.t += 1

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_vec = x.flatten()
        theta = self.A_inv @ self.b
        uncertainty = np.sqrt(max(0.0, np.dot(x_vec, self.A_inv @ x_vec)))
        expected_reward = np.dot(theta, x_vec)
        return np.array([expected_reward]), np.array([uncertainty])

    def sample(self, x: np.ndarray) -> np.ndarray:
        x_vec = x.flatten()
        nu_t = self.nu / np.sqrt(self.t + 1)
        theta_hat = self.A_inv @ self.b
        cov = (nu_t ** 2) * self.A_inv + 1e-6 * np.eye(self.feature_dim)
        try:
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
        except np.linalg.LinAlgError:
            theta_sample = theta_hat
        return np.array([np.dot(theta_sample, x_vec)])
