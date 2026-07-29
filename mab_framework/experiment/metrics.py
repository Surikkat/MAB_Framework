import numpy as np

class MetricsTracker:
    def __init__(self, n_steps: int = 0):
        if n_steps > 0:
            self.rewards = np.empty(n_steps)
            self.regrets = np.empty(n_steps)
            self.times = np.empty(n_steps)
            self._preallocated = True
        else:
            self.rewards = []
            self.regrets = []
            self.times = []
            self._preallocated = False
        self._idx = 0
        
    def add(self, reward: float, regret: float, iter_time: float):
        if self._preallocated:
            self.rewards[self._idx] = reward
            self.regrets[self._idx] = regret
            self.times[self._idx] = iter_time
            self._idx += 1
        else:
            self.rewards.append(reward)
            self.regrets.append(regret)
            self.times.append(iter_time)
        
    def get_metrics(self):
        rewards_arr = self.rewards[:self._idx] if self._preallocated else np.array(self.rewards)
        regrets_arr = self.regrets[:self._idx] if self._preallocated else np.array(self.regrets)
        times_arr = self.times[:self._idx] if self._preallocated else np.array(self.times)
        
        cum_rewards = np.cumsum(rewards_arr)
        cum_regrets = np.cumsum(regrets_arr)
        
        num_steps = np.arange(1, len(regrets_arr) + 1)
        avg_regrets = (cum_regrets / num_steps).tolist()
        
        return {
            "rewards": rewards_arr.tolist(),
            "cumulative_reward": cum_rewards.tolist(),
            "regrets": regrets_arr.tolist(),
            "cumulative_regret": cum_regrets.tolist(),
            "average_regret": avg_regrets,
            "times": times_arr.tolist()
        }
