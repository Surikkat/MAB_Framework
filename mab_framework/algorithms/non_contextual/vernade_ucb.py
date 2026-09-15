import numpy as np
from collections import deque
from mab_framework.algorithms.base import BaseAlgorithm

class VernadeDelayedUCB(BaseAlgorithm):
    """
    VernadeDelayedUCB
    
    Source: Vernade et al., 2017, "Stochastic Bandit Models for Delayed Conversions"
    (Eq. 5 and Section 5.2).
    """
    def __init__(self, n_arms, model=None, delay_cdf=None, D_max=1000, **kwargs):
        super().__init__(n_arms=n_arms, model=model)
        if delay_cdf is None:
            self.delay_cdf = lambda d: 1.0 - np.exp(-0.1 * d)
        else:
            self.delay_cdf = delay_cdf
            
        self.D_max = D_max
        # Храним активные таймстемпы в деке для O(1) амортизированного удаления
        self.active_timestamps = [deque() for _ in range(n_arms)]
        self.matured_pulls = np.zeros(n_arms)
        
        self.sum_observed_rewards = np.zeros(n_arms)
        self.total_pulls = np.zeros(n_arms)
        self.t = 1

    def select_arm(self, context=None):
        ucbs = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            if self.total_pulls[a] == 0:
                ucbs[a] = np.inf
            else:
                # Очистка старых пулов: если пул "созрел" (t - s >= D_max)
                queue = self.active_timestamps[a]
                while queue and (self.t - queue[0]) >= self.D_max:
                    queue.popleft()
                    self.matured_pulls[a] += 1
                
                # Векторизованное вычисление N_tilde для активных пулов
                if len(queue) == 0:
                    active_N_tilde = 0.0
                else:
                    valid_timestamps = np.array(queue)
                    active_N_tilde = np.sum(self.delay_cdf(self.t - valid_timestamps))
                        
                N_tilde = self.matured_pulls[a] + active_N_tilde
                
                # Защита от нулевого CDF
                if N_tilde <= 1e-9:
                    ucbs[a] = np.inf
                else:
                    theta_hat = self.sum_observed_rewards[a] / N_tilde
                    bonus = np.sqrt((self.total_pulls[a] / N_tilde) * (np.log(self.t) / (2 * N_tilde)))
                    ucbs[a] = theta_hat + bonus
                    
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
            
        self.active_timestamps[action].append(self.t)
        self.total_pulls[action] += 1
        self.t += 1
        return action

    def update(self, feedbacks):
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            self.sum_observed_rewards[action] += reward
