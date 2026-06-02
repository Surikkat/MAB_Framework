import numpy as np
from mab_framework.algorithms.base import BaseAlgorithm

class JoulaniDelayedUCB(BaseAlgorithm):
    """
    JoulaniDelayedUCB
    
    Source: Joulani et al., 2013, "Online Learning under Delayed Feedback"
    (Appendix B, Eq. 6 & Theorem 7).
    """
    def __init__(self, n_arms, **kwargs):
        # Strictly non-contextual algorithm
        super().__init__(n_arms=n_arms, model=None)
        self.model = None
        self.resolved_counts = np.zeros(n_arms)
        self.total_pulls = np.zeros(n_arms)
        self.emp_means = np.zeros(n_arms)
        self.t = 0

    def select_arm(self, context=None):
        self.t += 1
        ucbs = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            if self.resolved_counts[a] == 0:
                ucbs[a] = np.inf
            else:
                # Ограничение Хеффдинга вычисляется строго по разрешенным данным
                ucbs[a] = self.emp_means[a] + np.sqrt(2 * np.log(self.t) / self.resolved_counts[a])
                
        # Рандомизированный tie-breaking среди максимумов
        max_val = np.max(ucbs)
        max_indices = np.flatnonzero(ucbs == max_val)
        
        if np.isinf(max_val):
            # При бесконечных UCB (нехватка данных) отдаем приоритет наименее исследованным ручкам
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
            
            self.resolved_counts[action] += 1
            self.emp_means[action] += (reward - self.emp_means[action]) / self.resolved_counts[action]
