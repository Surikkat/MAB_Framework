import numpy as np
from .base import BaseModel


class GPRFFModel(BaseModel):
    def __init__(self, feature_dim: int, kernel_scale: float = 1.0, lengthscale: float = 1.0,
                 n_features: int = 500, lambda_prior: float = 1.0, nu0: float = 1.0):
        self.d = feature_dim
        self.scale = kernel_scale
        self.ls = lengthscale
        self.D = n_features
        self.lambda0 = lambda_prior
        self.nu0 = nu0
        self.W = np.random.normal(0, (2 * np.pi) / self.ls, size=(self.d, self.D))
        self.b = np.random.uniform(0, 2 * np.pi, size=self.D)
        self.A = self.lambda0 * np.eye(self.D)
        self.v = np.zeros(self.D)
        self.t = 0

    def _phi(self, x: np.ndarray) -> np.ndarray:
        proj = x.dot(self.W) + self.b
        return np.sqrt(2 * self.scale / self.D) * np.cos(proj)

    def fit(self, x: np.ndarray, y: float):
        phi = self._phi(x)
        self.A += np.outer(phi, phi)
        self.v += phi * y
        self.t += 1

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = self._phi(x)
        try:
            L = np.linalg.cholesky(self.A)
            mu = np.linalg.solve(L.T, np.linalg.solve(L, self.v))
            A_inv_phi = np.linalg.solve(L.T, np.linalg.solve(L, phi))
        except np.linalg.LinAlgError:
            A_inv = np.linalg.inv(self.A)
            mu = A_inv @ self.v
            A_inv_phi = A_inv @ phi
        expected_reward = float(phi @ mu)
        sigma = float(np.sqrt(np.abs(phi @ A_inv_phi)))
        return np.array([expected_reward]), np.array([sigma])

    def sample(self, x: np.ndarray) -> np.ndarray:
        phi = self._phi(x)
        nu_t = self.nu0 * np.sqrt(np.log(self.t + 2))
        try:
            L = np.linalg.cholesky(self.A)
            mu = np.linalg.solve(L.T, np.linalg.solve(L, self.v))
            eps = np.random.randn(self.D)
            theta = mu + nu_t * np.linalg.solve(L.T, eps)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.inv(self.A)
            mu = A_inv @ self.v
            cov = (nu_t ** 2) * A_inv
            theta = np.random.multivariate_normal(mu, cov)
        return np.array([float(phi @ theta)])
