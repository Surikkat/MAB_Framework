import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from .base import BaseEnvironment


class _PatchedOpenBanditDataset:
    def __init__(self, behavior_policy="random", campaign="all"):
        from obp.dataset import OpenBanditDataset as _OBD

        obd = object.__new__(_OBD)
        obd.behavior_policy = behavior_policy
        obd.campaign = campaign
        obd.data_path = None
        obd.dataset_name = "obd"

        if behavior_policy not in ("bts", "random"):
            raise ValueError(f"behavior_policy must be 'bts' or 'random', got {behavior_policy}")
        if campaign not in ("all", "men", "women"):
            raise ValueError(f"campaign must be 'all', 'men', or 'women', got {campaign}")

        obd.data_path = Path(_OBD.__module__.replace(".", "/")).parent.parent / "dataset" / "obd"
        import obp.dataset.real as _real_mod
        obd.data_path = Path(_real_mod.__file__).parent / "obd"
        obd.data_path = obd.data_path / behavior_policy / campaign
        obd.raw_data_file = f"{campaign}.csv"

        obd.load_raw_data()

        user_cols = obd.data.columns.str.contains("user_feature")
        obd.context = pd.get_dummies(
            obd.data.loc[:, user_cols], drop_first=True
        ).values
        item_feature_0 = obd.item_context["item_feature_0"]
        item_feature_cat = obd.item_context.drop(columns="item_feature_0").apply(
            LabelEncoder().fit_transform
        )
        obd.action_context = pd.concat([item_feature_cat, item_feature_0], axis=1).values

        self._obd = obd

    @property
    def n_actions(self):
        return int(self._obd.action.max() + 1)

    def obtain_batch_bandit_feedback(self):
        obd = self._obd
        return dict(
            n_rounds=obd.data.shape[0],
            n_actions=self.n_actions,
            action=obd.action,
            position=obd.position,
            reward=obd.reward,
            pscore=obd.pscore,
            context=obd.context,
            action_context=obd.action_context,
        )


class OpenBanditEnvironment(BaseEnvironment):
    def __init__(self, behavior_policy="random", campaign="all",
                 reward_mode="replay", max_steps=None, **kwargs):
        super().__init__(**kwargs)

        dataset = _PatchedOpenBanditDataset(
            behavior_policy=behavior_policy, campaign=campaign
        )
        feedback = dataset.obtain_batch_bandit_feedback()

        self.contexts = feedback["context"]
        self.logged_actions = feedback["action"]
        self.rewards = feedback["reward"]
        self.n_arms = feedback["n_actions"]
        self.reward_mode = reward_mode

        self.T = len(self.contexts)
        if max_steps is not None:
            self.T = min(self.T, max_steps)

        self.current_idx = 0

    def reset(self) -> None:
        self.current_idx = 0
        self.delay_buffer.queue.clear()
        self.delay_buffer.current_time = 0

    def get_context(self) -> np.ndarray:
        if self.current_idx >= self.T:
            raise IndexError(
                f"get_context() called at index {self.current_idx}, "
                f"but dataset has only {self.T} entries."
            )
        ctx = self.contexts[self.current_idx]
        return np.tile(ctx, (self.n_arms, 1))

    def _step_raw(self, action: int) -> tuple[float, float]:
        if self.current_idx >= self.T:
            raise IndexError(
                f"step() called at index {self.current_idx}, "
                f"but dataset has only {self.T} entries."
            )

        logged_action = int(self.logged_actions[self.current_idx])
        logged_reward = float(self.rewards[self.current_idx])
        self.current_idx += 1

        if self.reward_mode == "replay":
            if action == logged_action:
                return logged_reward, logged_reward
            else:
                return 0.0, 0.0
        else:
            if action == logged_action:
                return logged_reward, logged_reward
            else:
                return 0.0, logged_reward
