"""GP_UCB_KernelFlexible — exact reproduction of NN_AGP/bandits/gp_ucb_multikernel.py.

GP-UCB with multiplicative or adaptive kernel over (theta, x) pairs.
"""
import math
import numpy as np
from scipy.linalg import cho_solve, cho_factor
from .base import BaseAlgorithm


class GPUCBKernelFlexibleAlgorithm(BaseAlgorithm):
    """Exact port of NN_AGP/bandits/gp_ucb_multikernel.py GP_UCB_KernelFlexible.

    Single shared GP model over (theta, x) concatenated inputs.
    Expects context shape (n_arms, feature_dim) where each row is [theta_t, x_a].
    """
    def __init__(self, n_arms: int, x_dim: int, theta_dim: int,
                 sigma_noise: float = 0.01, beta: float = 2.0,
                 lengthscale_x: float = 1.0, lengthscale_theta: float = 1.0,
                 kernel_type: str = 'multiplicative',
                 adaptive_weights: dict = None,
                 model=None):
        super().__init__(n_arms, model)
        self.x_dim = x_dim
        self.theta_dim = theta_dim
        self.sigma_noise = sigma_noise
        self.beta = beta
        self.l_x = lengthscale_x
        self.l_theta = lengthscale_theta
        self.kernel_type = kernel_type

        if adaptive_weights is None:
            self.adaptive_weights = {'theta': 0.5, 'x': 0.5}
        else:
            self.adaptive_weights = adaptive_weights

        self.X = []
        self.y = []

    def rbf_kernel(self, X1, X2, lengthscale):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / lengthscale**2)

    def kernel(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        theta1, x1 = X1[:, :self.theta_dim], X1[:, self.theta_dim:]
        theta2, x2 = X2[:, :self.theta_dim], X2[:, self.theta_dim:]

        k_theta = self.rbf_kernel(theta1, theta2, self.l_theta)
        k_x = self.rbf_kernel(x1, x2, self.l_x)

        if self.kernel_type == 'multiplicative':
            return k_theta * k_x
        elif self.kernel_type == 'adaptive':
            w_theta = self.adaptive_weights.get('theta', 0.5)
            w_x = self.adaptive_weights.get('x', 0.5)
            return w_theta * k_theta + w_x * k_x

    def select_arm(self, context: np.ndarray) -> int:
        if context.ndim == 1:
            context = context.reshape(1, -1)

        inputs = context  # already (n_arms, theta_dim + x_dim)

        if len(self.X) == 0:
            return np.random.randint(context.shape[0])

        X_train = np.array(self.X)
        y_train = np.array(self.y)

        K = self.kernel(X_train, X_train) + self.sigma_noise**2 * np.eye(len(self.X))
        K_s = self.kernel(X_train, inputs)
        K_ss = self.kernel(inputs, inputs)

        L = cho_factor(K, lower=True)
        alpha = cho_solve(L, y_train)

        mu = K_s.T @ alpha
        v = cho_solve(L, K_s)
        sigma = np.sqrt(np.maximum(np.diag(K_ss - K_s.T @ v), 1e-10))

        ucb = mu + math.sqrt(self.beta) * sigma
        return int(np.argmax(ucb))

    def update(self, context: np.ndarray, action: int, reward: float):
        if context.ndim > 1:
            new_input = context[action]
        else:
            new_input = context
        self.X.append(new_input)
        self.y.append(reward)
