import numpy as np
from mab_framework.algorithms.base import BaseAlgorithm

class PatientBandits(BaseAlgorithm):
    """
    PatientBandits
    
    Source: Manegueu et al., 2020, "Stochastic bandits with arm-dependent delays"
    (Eq. 4 and 5).
    """
    def __init__(self, n_arms, model=None, alpha=0.5, horizon_T=2000, **kwargs):
        super().__init__(n_arms=n_arms, model=model)
        self.alpha_param = min(alpha, 0.5)
        # Обязательный параметр горизонта T для доказательства Finite-Horizon
        self.horizon_T = horizon_T
        self.total_pulls = np.zeros(n_arms)
        self.sum_observed_rewards = np.zeros(n_arms)

    def select_arm(self, context=None):
        ucbs = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            if self.total_pulls[a] == 0:
                ucbs[a] = np.inf
            else:
                mu_hat = self.sum_observed_rewards[a] / self.total_pulls[a]
                # Используем horizon_T вместо динамического t
                term2 = np.sqrt(2 * np.log(2 * self.n_arms * (self.horizon_T ** 3)) / self.total_pulls[a])
                term3 = 2 * (self.total_pulls[a] ** (-self.alpha_param))
                ucbs[a] = mu_hat + term2 + term3
                
        # Рандомизированный tie-breaking среди максимумов
        max_val = np.max(ucbs)
        max_indices = np.flatnonzero(ucbs == max_val)
        
        if np.isinf(max_val):
            pulls_of_max = self.total_pulls[max_indices]
            min_pulls = np.min(pulls_of_max)
            best_indices = max_indices[pulls_of_max == min_pulls]
            action = int(np.random.choice(best_indices))
        else:
            action = int(np.random.choice(max_indices))
            
        self.total_pulls[action] += 1
        return action

    def update(self, feedbacks):
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            self.sum_observed_rewards[action] += reward
