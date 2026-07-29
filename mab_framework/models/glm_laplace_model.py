import numpy as np
from scipy.special import expit
from .base import BaseModel


class GLMLaplaceModel(BaseModel):
    """
    Generalized Linear Model (GLM) with Laplace Approximation.
    
    References
    ----------
    Filippi, S., Cappe, O., Garivier, A., & Szepesvari, C. (2010). 
    "Parametric Bandits: The Generalized Linear Case." NeurIPS.
    """
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
            if x.ndim == 2:
                solved = np.linalg.solve(self.hessian, x.T)
                var = np.sum(x.T * solved, axis=0) * (self.alpha ** 2)
                sigma = np.sqrt(np.abs(var))
            else:
                solved = np.linalg.solve(self.hessian, x)
                var = (x @ solved) * (self.alpha ** 2)
                sigma = np.array([float(np.sqrt(np.abs(var)))])
                mu = np.array([float(mu)])
        except np.linalg.LinAlgError:
            if x.ndim == 2:
                sigma = np.full(x.shape[0], self.alpha)
            else:
                sigma = np.array([self.alpha])
                mu = np.array([float(mu)])
        return mu, sigma

    def sample(self, x: np.ndarray) -> np.ndarray:
        try:
            L = np.linalg.cholesky(self.hessian)
            z = np.random.standard_normal(self.d)
            theta_sample = self.theta_map + self.alpha * np.linalg.solve(L.T, z)
        except np.linalg.LinAlgError:
            theta_sample = self.theta_map
        mu = expit(x @ theta_sample)
        return np.array([float(mu)])
