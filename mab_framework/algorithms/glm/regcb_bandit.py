from sklearn.linear_model import SGDRegressor
from ..base import BaseAlgorithm
import numpy as np
from typing import List, Dict, Any


class RegCBBandit(BaseAlgorithm):
    """
    Regression-based Contextual Bandit (RegCB) Algorithm.

    A contextual bandit algorithm that reduces the problem to online regression.
    This implementation replaces the Vowpal Wabbit dependency with an ensemble
    of SGDRegressors from scikit-learn to estimate both the expected reward and
    the prediction uncertainty (confidence width), capturing the essence of the
    regression oracle approach.

    References
    ----------
    Foster, D. J., Agarwal, A., Dudik, M., & Schapire, R. E. (2018). 
    "Practical Contextual Bandits with Regression Oracles." ICML.
    """
    def __init__(self,
                 n_arms: int,
                 context_dim: int,
                 model=None,
                 alpha: float = 1.0,
                 n_estimators: int = 3
                 ):
        super().__init__(n_arms, model)
        self.actions = list(range(n_arms))
        self.d = context_dim
        self.alpha = alpha
        self.n_estimators = n_estimators
        
        # Initialize an ensemble of regressors for each arm
        self.models = {
            arm: [SGDRegressor(learning_rate='constant', eta0=0.01) for _ in range(n_estimators)]
            for arm in self.actions
        }
        self.is_fitted = {arm: [False] * n_estimators for arm in self.actions}
        self.action_probs = None
    
    def select_arm(self, context: np.ndarray) -> int:
        # context shape: (n_arms, context_dim) or (context_dim,)
        if context.ndim == 1:
            context = np.tile(context, (self.n_arms, 1))
            
        ucbs = np.zeros(self.n_arms)
        
        for arm in self.actions:
            ctx = context[arm].reshape(1, -1)
            
            preds = []
            for i in range(self.n_estimators):
                if self.is_fitted[arm][i]:
                    preds.append(self.models[arm][i].predict(ctx)[0])
                else:
                    preds.append(0.0)
                    
            if not any(self.is_fitted[arm]):
                ucbs[arm] = np.inf
            else:
                mean_pred = np.mean(preds)
                std_pred = np.std(preds) if len(preds) > 1 else 1.0
                ucbs[arm] = mean_pred + self.alpha * std_pred
                
        # Select arm with max UCB
        best_arm = int(np.argmax(ucbs))
        
        # Fallback to random if all infinity (untrained)
        if np.isinf(ucbs[best_arm]):
            unexplored = [a for a in self.actions if np.isinf(ucbs[a])]
            best_arm = int(np.random.choice(unexplored))
            
        self.action_probs = [1.0 if a == best_arm else 0.0 for a in self.actions]
        return best_arm
    
    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = float(fb["reward"])
            context = fb["context"]
            arm = int(action)
            
            ctx = np.asarray(context, dtype=float)
            if ctx.ndim == 2:
                ctx = ctx[arm].reshape(1, -1)
            else:
                ctx = ctx.flatten().reshape(1, -1)
            
            # Online update for the ensemble with bootstrap-like random weights
            for i in range(self.n_estimators):
                weight = np.random.poisson(1.0)
                for _ in range(weight):
                    self.models[arm][i].partial_fit(ctx, [reward])
                if weight > 0:
                    self.is_fitted[arm][i] = True
