import numpy as np
from .base import BaseAlgorithm
from scipy.special import expit


class SGDTSBandit(BaseAlgorithm):
    def __init__(self, d, K=2, nu=0.1, lr=0.01, lambda_prior=1.0, warmup_steps=50, mle_lr:float =0.1, mle_steps: int = 500, model=None):
        super().__init__(K, model)
        self.original_d = d  # dimensionality of a single arm context
        self.d = K * d  # Total input dimension after expansion
        self.nu = nu  # Exploration factor
        self.lr = lr  # Learning rate
        self.lambda_prior = lambda_prior
        self.warmup_steps = warmup_steps  # Tau
        self.mle_lr = mle_lr
        self.mle_steps = mle_steps
        self.reset()

    def reset(self):
        self.theta = np.zeros(self.d)
        self.V_diag = self.lambda_prior * np.ones(self.d)  # diag of Gram matrix (ridge prior)
        self.t = 0
        self.X_buffer = []                             # store features for MLE
        self.y_buffer = []  

    def _expand_context(self, context):
        """
        Expand from shape (K, d) → (K, K*d) by placing each arm's context in its slot
        """
        expanded = np.zeros((self.n_arms, self.d))
        for k in range(self.n_arms):
            start = k * self.original_d
            end = (k + 1) * self.original_d
            expanded[k, start:end] = context[k]
        return expanded

    def mu_fucntion(self, z):
        return expit(z)

    def select_arm(self, context: np.ndarray) -> int:
        expanded_context = self._expand_context(context)

        std = 1.0 / np.sqrt(self.V_diag)  # shape: (d, 1)
        noise = self.nu * np.random.randn(self.d) * std  # shape: 
        theta_sample = self.theta + noise
        scores = expanded_context.dot(theta_sample)
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, arm: int, reward: float):
        x = np.zeros(self.d)  # shape (K*d,)
        start = arm * self.original_d
        end = (arm + 1) * self.original_d
        x[start:end] = context  # set up context only for specific arm

        # Cache context and target for buffer
        if self.t < self.warmup_steps:
            self.X_buffer.append(x)
            self.y_buffer.append(reward)

            if self.t == self.warmup_steps - 1:
                # Perform ridge regression MLE at warmup_steps
                X = np.vstack(self.X_buffer)  # shape: (warmup_steps, K*d)
                y = np.array(self.y_buffer)  # shape: (warmup_steps,)
                theta = np.zeros(self.d)  # learning parameter vector with dimension (K*d,) for each arm
                for _ in range(self.mle_steps):
                    mu_res = self.mu_fucntion(X.dot(theta))
                    loss_grad = X.T.dot(mu_res - y) + self.lambda_prior * theta  # L2 - regularization

                    theta -= self.mle_lr * loss_grad

                self.theta = theta
                self.V_diag += np.sum(X ** 2, axis=0)  # accumulate V_diag from warmup data
        else:
            # Online SGD update
            mu_res = self.mu_fucntion(self.theta.dot(x))
            sgd_loss_grad = (mu_res - reward) * x + self.lambda_prior * self.theta
            self.theta -= self.lr * sgd_loss_grad

            # Update V_diag
            self.V_diag += x ** 2
        self.t += 1
