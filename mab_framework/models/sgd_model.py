import numpy as np
from scipy.special import expit
from .base import BaseModel

class SGDModel(BaseModel):
    def __init__(self, feature_dim: int, nu: float = 0.1, lr: float = 0.01, lambda_prior: float = 1.0, warmup_steps: int = 50, mle_lr: float = 0.1, mle_steps: int = 500):
        self.d = feature_dim
        self.nu = nu
        self.lr = lr
        self.lambda_prior = lambda_prior
        self.warmup_steps = warmup_steps
        self.mle_lr = mle_lr
        self.mle_steps = mle_steps
        self.reset()

    def reset(self):
        self.theta = np.zeros(self.d)
        self.V_diag = self.lambda_prior * np.ones(self.d)
        self.t = 0
        self.X_buffer = []
        self.y_buffer = []

    def mu_function(self, z):
        return expit(z)

    def fit(self, x: np.ndarray, y: float) -> None:
        if self.t < self.warmup_steps:
            self.X_buffer.append(x)
            self.y_buffer.append(y)

            if self.t == self.warmup_steps - 1:
                X = np.vstack(self.X_buffer)
                y_arr = np.array(self.y_buffer)
                theta = np.zeros(self.d)
                for _ in range(self.mle_steps):
                    mu_res = self.mu_function(X.dot(theta))
                    loss_grad = X.T.dot(mu_res - y_arr) + self.lambda_prior * theta
                    theta -= self.mle_lr * loss_grad

                self.theta = theta
                self.V_diag += np.sum(X ** 2, axis=0)
        else:
            mu_res = self.mu_function(self.theta.dot(x))
            sgd_loss_grad = (mu_res - y) * x + self.lambda_prior * self.theta
            self.theta -= self.lr * sgd_loss_grad
            self.V_diag += x ** 2
            
        self.t += 1

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = x.dot(self.theta)
        std = 1.0 / np.sqrt(self.V_diag)
        sigma = np.sqrt(np.sum((x * std)**2, axis=-1)) if x.ndim > 1 else np.sqrt(np.sum((x * std)**2))
        return np.array([mu]), np.array([sigma])

    def sample(self, x: np.ndarray) -> np.ndarray:
        std = 1.0 / np.sqrt(self.V_diag)
        noise = self.nu * np.random.randn(self.d) * std
        theta_sample = self.theta + noise
        if x.ndim == 1:
            return np.array([x.dot(theta_sample)])
        return x.dot(theta_sample)
