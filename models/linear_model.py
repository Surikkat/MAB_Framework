import numpy as np
from .base import BaseModel

class OnlineRidgeRegression(BaseModel):
    def __init__(self, feature_dim: int, l2_reg: float = 1.0, nu: float = 1.0):
        self.feature_dim = feature_dim
        self.nu = nu
        self.b = np.zeros(feature_dim)
        self.A_inv = np.eye(feature_dim) / l2_reg
        self.t = 0

    def fit(self, x: np.ndarray, y: float):
        x = x.reshape(-1, 1)
        self.A_inv -= (self.A_inv @ x @ x.T @ self.A_inv) / (1 + x.T @ self.A_inv @ x)
        self.b += y * x.flatten()
        self.t += 1

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = self.A_inv @ self.b
        expected_reward = np.dot(theta, x)
        uncertainty = np.sqrt(x.T @ self.A_inv @ x)
        return np.array([expected_reward]), np.array([uncertainty])

    def sample(self, x: np.ndarray) -> np.ndarray:
        theta_hat = self.A_inv @ self.b
        nu_t = self.nu / np.sqrt(self.t + 1)
        cov = (nu_t ** 2) * self.A_inv + 1e-6 * np.eye(self.feature_dim)
        try:
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
        except np.linalg.LinAlgError:
            theta_sample = theta_hat
        return np.array([np.dot(theta_sample, x)])
