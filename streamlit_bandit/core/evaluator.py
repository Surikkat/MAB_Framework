import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class OPEEvaluator:
    def __init__(self, df_log, clipping_value=10, min_propensity=0.01, methods=['dm', 'ips', 'dr']):
        self.df = df_log
        self.clipping_value = clipping_value
        self.min_propensity = min_propensity
        self.methods = methods

        self.reward_model = None
        self._train_reward_model()
        
    def _train_reward_model(self):
        feature_cols = ['user_age', 'user_activity_score', 'hour_of_day', 
                       'item_price', 'item_rating']

        available_cols = [c for c in feature_cols if c in self.df.columns]
        
        categorical_cols = ['user_gender', 'device_type', 'category_id']
        for col in categorical_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                available_cols.extend(dummies.columns.tolist())
        
        X = self.df[available_cols].fillna(0)
        y = self.df['reward']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        self.reward_model = GradientBoostingClassifier(
            n_estimators=50, max_depth=4, random_state=42
        )
        self.reward_model.fit(X_train, y_train)

        self.reward_auc = roc_auc_score(y_test, self.reward_model.predict_proba(X_test)[:, 1])

        self.feature_cols = available_cols

        self.predicted_rewards = self.reward_model.predict_proba(X)[:, 1]
        
    def evaluate(self, candidate_name, candidate_fn):
        result = {'candidate': candidate_name}

        pi_e = candidate_fn(self.df)

        pi_b = np.clip(self.df['propensity'].values, self.min_propensity, 1.0)

        weights = np.where(pi_b > 0, pi_e / pi_b, 0)
        weights_clipped = np.clip(weights, 0, self.clipping_value)
        
        # 1. Direct Method
        if 'dm' in self.methods:
            result['dm_score'] = np.mean(pi_e * self.predicted_rewards)
        
        # 2. IPS
        if 'ips' in self.methods:
            result['ips_score'] = np.mean(weights_clipped * self.df['reward'].values)
        
        # 3. Doubly Robust
        if 'dr' in self.methods:
            residuals = self.df['reward'].values - self.predicted_rewards
            dr_correction = np.mean(weights_clipped * residuals)
            result['dr_score'] = result.get('dm_score', 0) + dr_correction

        if 'dr_score' not in result:
            result['dr_score'] = result.get('dm_score', result.get('ips_score', 0))

        result['effective_sample_size'] = weights.sum()**2 / (weights**2).sum() if weights.sum() > 0 else 0

        result['max_weight'] = weights.max()
        result['mean_weight'] = weights.mean()
        
        return result