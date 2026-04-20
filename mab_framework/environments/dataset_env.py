import os
import warnings
import numpy as np
import pandas as pd
from .base import BaseEnvironment

class DatasetEnvironment(BaseEnvironment):
    def __init__(self, dataset_path: str, max_steps: int = None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_path = dataset_path
        self.current_step = 0
        self.max_steps = max_steps
        self._load_dataset()

    def _load_dataset(self):
        if self.dataset_path.endswith('.npz'):
            data = np.load(self.dataset_path, allow_pickle=True)
            self.rewards = data['rewards_clean']
            self.T = int(data['T'])
            self.n_arms = int(data['pool_size'])
            self.X_pool = data['X_pool']

            if 'thetas' in data:
                self.thetas = data['thetas']
                self.dynamic_context = True
                self.context_mode = 'theta_x'
                self.theta_dim = int(data['theta_dim']) if 'theta_dim' in data else int(self.thetas.shape[1])
            else:
                self.contexts = self.X_pool
                self.thetas = None
                self.dynamic_context = False
                self.context_mode = 'x_only'
                warnings.warn(
                    f"Dataset '{self.dataset_path}' has no 'thetas'. Using 'x_only' mode: "
                    f"context is static X_pool repeated every step. "
                    f"If rewards vary by step, the model won't capture that variation from context alone.",
                    UserWarning,
                    stacklevel=2
                )

        elif self.dataset_path.endswith('.csv'):
            df = pd.read_csv(self.dataset_path)
            ctx_cols = sorted([c for c in df.columns if c.startswith('context_')])
            self.n_arms = df['arm'].nunique()
            self.T = df['t'].nunique()

            contexts_all = []
            rewards_all = []
            for t_val in sorted(df['t'].unique()):
                t_df = df[df['t'] == t_val].sort_values('arm')
                contexts_all.append(t_df[ctx_cols].values)
                rewards_all.append(t_df['reward'].values)

            self.contexts_per_step = np.array(contexts_all)
            self.rewards = np.array(rewards_all)
            self.dynamic_context = True
            self.context_mode = 'per_arm_per_step'
            self.thetas = None
            self.X_pool = None

        elif os.path.isdir(self.dataset_path):
            x_path = os.path.join(self.dataset_path, "X.csv")
            y_path = os.path.join(self.dataset_path, "Y.csv")
            self.contexts = pd.read_csv(x_path).values
            self.rewards = pd.read_csv(y_path).values
            self.T = self.rewards.shape[0]
            self.n_arms = self.rewards.shape[1]
            self.dynamic_context = True
            self.context_mode = 'flat'
            self.thetas = None
            self.X_pool = None

        if self.max_steps is not None:
            self.T = min(self.T, self.max_steps)

    def reset(self) -> None:
        self.current_step = 0

    def get_context(self) -> np.ndarray:
        if self.current_step >= self.T:
            raise IndexError(
                f"get_context() called at step {self.current_step}, but dataset has only {self.T} steps."
            )
        if self.context_mode == 'theta_x':
            theta_t = self.thetas[self.current_step]
            context = np.array([
                np.concatenate([theta_t, self.X_pool[a]])
                for a in range(self.n_arms)
            ])
            return context
        elif self.context_mode == 'per_arm_per_step':
            return self.contexts_per_step[self.current_step]
        elif self.context_mode == 'flat':
            context_flat = self.contexts[self.current_step]
            return np.tile(context_flat, (self.n_arms, 1))
        else:
            return self.contexts

    def _step_raw(self, action: int) -> tuple[float, float]:
        if self.current_step >= self.T:
            raise IndexError(
                f"step() called at step {self.current_step}, but dataset has only {self.T} steps."
            )
        if not (0 <= action < self.n_arms):
            raise ValueError(
                f"Invalid action {action}. Must be in [0, {self.n_arms})."
            )
        reward = float(self.rewards[self.current_step, action])
        optimal_reward = float(np.max(self.rewards[self.current_step]))
        self.current_step += 1
        return reward, optimal_reward