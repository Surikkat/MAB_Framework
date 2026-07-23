import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class RewardPredictor:
    def __init__(self, max_iter=200, learning_rate=0.1, max_depth=6):
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )

    def fit(self, contexts, actions, rewards):
        features = np.column_stack([contexts, actions.reshape(-1, 1)])
        self.model.fit(features, rewards.astype(int))

    def predict(self, contexts, n_actions):
        n_rounds = len(contexts)
        estimated_rewards = np.zeros((n_rounds, n_actions, 1))
        for a in range(n_actions):
            features = np.column_stack([contexts, np.full((n_rounds, 1), a)])
            proba = self.model.predict_proba(features)
            estimated_rewards[:, a, 0] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        return estimated_rewards
