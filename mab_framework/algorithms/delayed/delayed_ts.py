import numpy as np
from mab_framework.algorithms.base import BaseAlgorithm

class DelayedThompsonSampling(BaseAlgorithm):
    def __init__(self, n_arms, model, nu=1.0, gamma=1.0, **kwargs):
        super().__init__(n_arms=n_arms, model=model)
        self.nu = nu
        self.gamma = gamma
        self.pending_counts = np.zeros(n_arms)

    def select_arm(self, context=None):
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            x = context[a] if (isinstance(context, np.ndarray) and context.ndim == 2) else context
            mu_arr, sigma_arr = self.model[a].predict(x)
            
            mu = np.atleast_1d(mu_arr)[0]
            sigma = np.atleast_1d(sigma_arr)[0]
            
            # Строгое сужение (deflation) дисперсии
            var = sigma ** 2
            adjusted_var = var / (1.0 + self.gamma * self.pending_counts[a] * var)
            adjusted_sigma = np.sqrt(adjusted_var)
            
            # Сэмпл из нормального распределения
            scores[a] = np.random.normal(mu, adjusted_sigma * self.nu)
            
        action = int(np.argmax(scores))
        self.pending_counts[action] += 1
        return action

    def update(self, feedbacks):
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context_data = fb["context"]
            
            x = context_data[action] if (isinstance(context_data, np.ndarray) and context_data.ndim == 2) else context_data
            
            self.model[action].fit(x, reward)
            self.pending_counts[action] = max(0, self.pending_counts[action] - 1)
