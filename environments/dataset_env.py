import os
import numpy as np
import pandas as pd
from .base import BaseEnvironment

class DatasetEnvironment(BaseEnvironment):
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.current_step = 0
        self._load_dataset()

    def _load_dataset(self):
        if self.dataset_path.endswith('.npz'):
            data = np.load(self.dataset_path, allow_pickle=True)
            self.contexts = data['X_pool']
            self.rewards = data['rewards_clean']
            
            self.T = int(data['T'])
            self.n_arms = int(data['pool_size'])
            self.dynamic_context = False

        elif os.path.isdir(self.dataset_path):
            x_path = os.path.join(self.dataset_path, "X.csv")
            y_path = os.path.join(self.dataset_path, "Y.csv")

            self.contexts = pd.read_csv(x_path).values
            self.rewards = pd.read_csv(y_path).values
            
            self.T = self.rewards.shape[0]
            self.n_arms = self.rewards.shape[1]
            self.dynamic_context = True


    def reset(self) -> None:
        self.current_step = 0

    def get_context(self) -> np.ndarray:
        if self.dynamic_context:
            return self.contexts[self.current_step]
        else:
            return self.contexts

    def step(self, action: int) -> tuple[float, float]:
        reward = float(self.rewards[self.current_step, action])
        optimal_reward = float(np.max(self.rewards[self.current_step]))
        self.current_step += 1
        return reward, optimal_reward