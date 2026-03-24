import numpy as np
from .base import BaseModel

class OnlineRidgeRegression(BaseModel):
    def __init__(self, feature_dim: int, l2_reg: float = 1.0):
        self.feature_dim = feature_dim
        self.A = np.eye(feature_dim) * l2_reg
        self.b = np.zeros(feature_dim)
        self.A_inv = np.linalg.inv(self.A)

    def fit(self, x: np.ndarray, y: float):
        x = x.reshape(-1, 1)
        self.A_inv -= (self.A_inv @ x @ x.T @ self.A_inv) / (1 + x.T @ self.A_inv @ x)
        self.b += y * x.flatten()

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        theta = self.A_inv @ self.b
        expected_reward = np.dot(theta, x)
        uncertainty = np.sqrt(x.T @ self.A_inv @ x)
        
        return expected_reward, uncertainty
