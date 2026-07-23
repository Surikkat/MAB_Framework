import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

_root_dir = str(Path(__file__).resolve().parent.parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from mab_framework.experiment.ope_evaluator import OPEEvaluator as FrameworkOPEEvaluator

class OPEEvaluator:
    def __init__(self, df_log, clipping_value=10, min_propensity=0.01, methods=['dm', 'ips', 'dr']):
        self.df = df_log.copy()
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
        pscore = np.clip(self.df['propensity'].values, self.min_propensity, 1.0)

        feedback = {
            "action": np.zeros(len(self.df), dtype=int),
            "reward": self.df['reward'].values,
            "pscore": pscore,
        }
        
        eval_methods = [m.lower() for m in self.methods if m.lower() in {"dm", "ips", "dr", "snips"}]
        if not eval_methods:
            eval_methods = ["ips", "dm", "dr"]

        fw_evaluator = FrameworkOPEEvaluator(eval_methods, clipping_value=self.clipping_value, use_obp=False)
        res = fw_evaluator.evaluate(feedback, action_dist=pi_e, estimated_rewards_by_reg_model=self.predicted_rewards)

        if 'dm' in self.methods:
            result['dm_score'] = res.get('dm', 0.0)
        if 'ips' in self.methods:
            result['ips_score'] = res.get('ips', 0.0)
        if 'dr' in self.methods:
            result['dr_score'] = res.get('dr', result.get('dm_score', result.get('ips_score', 0.0)))
        if 'snips' in res:
            result['snips_score'] = res.get('snips', 0.0)

        result['effective_sample_size'] = res.get('ess', 0.0)
        result['max_weight'] = res.get('max_weight', 0.0)
        result['mean_weight'] = res.get('mean_weight', 0.0)
        
        return result