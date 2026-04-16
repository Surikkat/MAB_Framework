import numpy as np
from sklearn.linear_model import Lasso
from .base import BaseModel

class FGTSLassoModel(BaseModel):
    def __init__(self, feature_dim: int, lasso_alpha=None, lasso_start=100, lasso_period=100, window=500, sigma_noise=0.1, sigma_prior=1.0):
        self.feature_dim = feature_dim
        self.lasso_alpha = lasso_alpha
        self.lasso_start = lasso_start
        self.lasso_period = lasso_period
        self.window = window
        self.sigma_noise = sigma_noise
        self.sigma_prior = sigma_prior
        self.t = 0
        self.X_hist = []
        self.Y_hist = []
        self.active_set = set(range(feature_dim))
        self.mu = np.zeros(feature_dim)
        self.Sigma = np.eye(feature_dim) * sigma_prior**2
        self.active_sets_history = []

    def fit(self, x: np.ndarray, y: float):
        self.t += 1
        self.X_hist.append(x)
        self.Y_hist.append(y)
        if len(self.X_hist) > self.window:
            self.X_hist = self.X_hist[-self.window:]
            self.Y_hist = self.Y_hist[-self.window:]
        if self.t >= self.lasso_start and self.t % self.lasso_period == 0:
            X_a = np.vstack(self.X_hist)
            y_a = np.array(self.Y_hist)
            alpha = self.lasso_alpha or np.sqrt(np.log(self.feature_dim) / max(1, X_a.shape[0]))
            lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
            lasso.fit(X_a, y_a)
            self.active_set = set(np.where(np.abs(lasso.coef_) > 1e-8)[0])
        idx = list(self.active_set)
        if idx:
            X_sub = np.vstack(self.X_hist)[:, idx]
            y_vec = np.array(self.Y_hist)
            Sigma_inv = np.linalg.inv(self.Sigma[np.ix_(idx, idx)])
            Sigma_new = np.linalg.inv(Sigma_inv + (1/self.sigma_noise**2) * X_sub.T @ X_sub)
            mu_new = Sigma_new @ (Sigma_inv @ self.mu[idx] + (1/self.sigma_noise**2) * X_sub.T @ y_vec)
            self.mu[idx] = mu_new
            self.Sigma[np.ix_(idx, idx)] = Sigma_new

        self.active_sets_history.append(set(self.active_set))

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = list(self.active_set)
        if not idx:
            return np.array([0.0]), np.array([0.0])
        mu_at_x = np.dot(x[idx], self.mu[idx])
        sigma_at_x = np.sqrt(x[idx].T @ self.Sigma[np.ix_(idx, idx)] @ x[idx])
        return np.array([mu_at_x]), np.array([sigma_at_x])

    def sample(self, x: np.ndarray) -> np.ndarray:
        idx = list(self.active_set)
        if not idx:
            return np.array([0.0])
        mu_sub = self.mu[idx]
        Sigma_sub = self.Sigma[np.ix_(idx, idx)]
        theta_sample = np.random.multivariate_normal(mu_sub, Sigma_sub)
        return np.array([np.dot(x[idx], theta_sample)])
