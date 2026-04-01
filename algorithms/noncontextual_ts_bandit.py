import numpy as np
from .base import BaseAlgorithm


class NonContextualTSBandit(BaseAlgorithm):
    def __init__(self, n_arms, prior_mean=0.0, prior_var=1.0, reward_var=1.0, model=None):
        super().__init__(n_arms, model)
        self.mu = np.ones(n_arms) * prior_mean
        self.lambda_ = np.ones(n_arms) / prior_var
        self.reward_var = reward_var
        self.counts = np.zeros(n_arms)

    def select_arm(self, context=None):
        std = np.sqrt(1 / self.lambda_)
        samples = np.random.normal(self.mu, std)
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        self.counts[arm] += 1
        self.lambda_[arm] += 1 / self.reward_var
        self.mu[arm] = (self.mu[arm] * (self.lambda_[arm] - 1 / self.reward_var) + reward / self.reward_var) / self.lambda_[arm]
