import numpy as np

class MetricsTracker:
    def __init__(self):
        self.rewards = []
        self.regrets = []
        self.times = []
        
    def add(self, reward: float, regret: float, iter_time: float):
        self.rewards.append(reward)
        self.regrets.append(regret)
        self.times.append(iter_time)
        
    def get_metrics(self):
        cum_rewards = np.cumsum(self.rewards).tolist()
        cum_regrets = np.cumsum(self.regrets).tolist()
        num_steps = np.arange(1, len(self.regrets) + 1)
        avg_regrets = (np.cumsum(self.regrets) / num_steps).tolist()
        return {
            "rewards": self.rewards,
            "cumulative_reward": cum_rewards,
            "regrets": self.regrets,
            "cumulative_regret": cum_regrets,
            "average_regret": avg_regrets,
            "times": self.times
        }
