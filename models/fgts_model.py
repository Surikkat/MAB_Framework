import numpy as np
from .base import BaseModel

class FGTSModel(BaseModel):
    def __init__(self, feature_dim: int, sigma_prior=1.0, sigma_noise=0.1, max_active=20):
        self.feature_dim = feature_dim
        self.sigma_prior = sigma_prior
        self.sigma_noise = sigma_noise
        self.max_active = max_active
        self.t = 0
        self.mu = np.zeros(feature_dim)
        self.Sigma = np.eye(feature_dim) * sigma_prior**2
        self.active_set = set(range(feature_dim))
        self.active_sets_history = []

    def fit(self, x: np.ndarray, y: float):
        self.t += 1
        Sigma_inv = np.linalg.inv(self.Sigma)
        Sigma_new = np.linalg.inv(Sigma_inv + (1/self.sigma_noise**2) * np.outer(x, x))
        mu_new = Sigma_new @ (Sigma_inv @ self.mu + (1/self.sigma_noise**2) * y * x)
        self.mu = mu_new
        self.Sigma = Sigma_new

        variances = np.diag(Sigma_new)
        score_blr = np.abs(mu_new) / (np.sqrt(variances) + 1e-12)
        top_idx = np.argsort(score_blr)[::-1][:self.max_active]
        self.active_set = set(top_idx)

        self.active_sets_history.append(set(self.active_set))

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu_at_x = np.dot(x, self.mu)
        sigma_at_x = np.sqrt(x.T @ self.Sigma @ x)
        return np.array([mu_at_x]), np.array([sigma_at_x])

    def sample(self, x: np.ndarray) -> np.ndarray:
        theta_sample = np.random.multivariate_normal(self.mu, self.Sigma)
        return np.array([np.dot(x, theta_sample)])
