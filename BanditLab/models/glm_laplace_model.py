import numpy as np
from scipy.special import expit
from .base import BaseModel


class GLMLaplaceModel(BaseModel):
    def __init__(self, feature_dim: int, prior_var: float = 1.0, lr: float = 0.1, alpha: float = 1.0):
        self.d = feature_dim
        self.prior_var = prior_var
        self.lr = lr
        self.alpha = alpha
        self.theta_map = np.zeros(self.d)
        self.hessian = np.eye(self.d) / prior_var

    def fit(self, x: np.ndarray, y: float):
        mu = expit(x @ self.theta_map)
        grad = x * (y - mu) - self.theta_map / self.prior_var
        W = mu * (1 - mu)
        hessian_update = np.outer(x, x) * W + np.eye(self.d) / self.prior_var
        self.theta_map += self.lr * grad
        self.hessian += hessian_update

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = expit(x @ self.theta_map)
        try:
            cov = np.linalg.inv(self.hessian) * (self.alpha ** 2)
            sigma = float(np.sqrt(np.abs(x @ cov @ x)))
        except np.linalg.LinAlgError:
            sigma = self.alpha
        return np.array([float(mu)]), np.array([sigma])

    def sample(self, x: np.ndarray) -> np.ndarray:
        try:
            cov = np.linalg.inv(self.hessian) * (self.alpha ** 2)
            theta_sample = np.random.multivariate_normal(self.theta_map, cov)
        except np.linalg.LinAlgError:
            theta_sample = self.theta_map
        mu = expit(x @ theta_sample)
        return np.array([float(mu)])
