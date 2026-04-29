import numpy as np
from .base import BaseModel

class OnlineRidgeRegression(BaseModel):
    def __init__(self, feature_dim: int, l2_reg: float = 1.0, nu: float = 1.0):
        self.feature_dim = feature_dim
        self.nu = nu
        self.l2_reg = l2_reg
        self.b = np.zeros(feature_dim)
        self.A = np.eye(feature_dim) * l2_reg
        self.t = 0

    def fit(self, x: np.ndarray, y: float):
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += y * x.flatten()
        self.t += 1

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            L = np.linalg.cholesky(self.A)
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b))
            v = np.linalg.solve(L, x)
            uncertainty = np.sqrt(v.T @ v)
        except np.linalg.LinAlgError:
            theta = np.linalg.solve(self.A, self.b)
            uncertainty = np.sqrt(x.T @ np.linalg.solve(self.A, x))
            
        expected_reward = np.dot(theta, x)
        return np.array([expected_reward]), np.array([uncertainty])

    def sample(self, x: np.ndarray) -> np.ndarray:
        try:
            L = np.linalg.cholesky(self.A)
            theta_hat = np.linalg.solve(L.T, np.linalg.solve(L, self.b))
            A_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(self.feature_dim)))
        except np.linalg.LinAlgError:
            theta_hat = np.linalg.solve(self.A, self.b)
            A_inv = np.linalg.solve(self.A, np.eye(self.feature_dim))
            
        nu_t = self.nu / np.sqrt(self.t + 1)
        cov = (nu_t ** 2) * A_inv + 1e-6 * np.eye(self.feature_dim)
        try:
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
        except np.linalg.LinAlgError:
            theta_sample = theta_hat
        return np.array([np.dot(theta_sample, x)])
