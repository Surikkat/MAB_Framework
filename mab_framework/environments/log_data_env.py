import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseEnvironment


SUPPORTED_FORMATS = {
    ".csv": pd.read_csv,
    ".json": pd.read_json,
    ".jsonl": lambda p: pd.read_json(p, lines=True),
    ".parquet": pd.read_parquet,
}


class LogDataEnvironment(BaseEnvironment):
    def __init__(self, log_path, context_columns="auto", action_column="action",
                 reward_column="reward", pscore_column="pscore",
                 n_actions=None, n_actions_column=None,
                 max_steps=None, **kwargs):
        super().__init__(**kwargs)

        path = Path(log_path)
        reader = SUPPORTED_FORMATS.get(path.suffix.lower())
        if reader is None:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {list(SUPPORTED_FORMATS.keys())}"
            )

        df = reader(path)

        if action_column not in df.columns and "item_id" in df.columns:
            action_column = "item_id"
        if pscore_column not in df.columns and "propensity" in df.columns:
            pscore_column = "propensity"

        self.logged_actions = df[action_column].values
        self.rewards = df[reward_column].values.astype(float)
        self.pscore = df[pscore_column].values.astype(float)

        exclude = {action_column, reward_column, pscore_column}
        if n_actions_column:
            exclude.add(n_actions_column)
        if "n_actions" in df.columns:
            exclude.add("n_actions")

        if context_columns == "auto":
            ctx_cols = [c for c in df.columns if c not in exclude]
        elif isinstance(context_columns, list):
            ctx_cols = context_columns
        else:
            raise ValueError("context_columns must be 'auto' or a list of column names")

        ctx_df = pd.get_dummies(df[ctx_cols], drop_first=True).astype(float)
        self.contexts = ctx_df.values

        if n_actions is not None:
            self.n_arms = int(n_actions)
        elif n_actions_column and n_actions_column in df.columns:
            self.n_arms = int(df[n_actions_column].max())
        elif "n_actions" in df.columns:
            self.n_arms = int(df["n_actions"].max())
        else:
            self.n_arms = int(self.logged_actions.max() + 1)

        self.T = len(self.contexts)
        if max_steps is not None:
            self.T = min(self.T, max_steps)
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        self.delay_buffer.queue.clear()
        self.delay_buffer.current_time = 0

    def get_context(self):
        if self.current_idx >= self.T:
            raise IndexError("End of dataset reached")
        return np.tile(self.contexts[self.current_idx], (self.n_arms, 1))

    def _step_raw(self, action):
        if self.current_idx >= self.T:
            raise IndexError("End of dataset reached")
        logged_action = int(self.logged_actions[self.current_idx])
        logged_reward = float(self.rewards[self.current_idx])
        self.current_idx += 1
        if action == logged_action:
            return logged_reward, logged_reward
        return 0.0, 0.0
