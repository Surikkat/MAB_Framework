import numpy as np
from sklearn.linear_model import Lasso
from .base import BaseModel

class FGTSModel(BaseModel):
    def __init__(self, feature_dim: int, lasso_alpha=None, lasso_start=100, lasso_period=100, window=500):
        self.feature_dim = feature_dim
        self.lasso_alpha = lasso_alpha
        self.lasso_start = lasso_start
        self.lasso_period = lasso_period
        self.window = window
        
        self.t = 0
        self.X_hist =[]
        self.Y_hist =[]
        
        self.active_set = set(range(feature_dim))
        self.mu = np.zeros(feature_dim)

    def fit(self, x: np.ndarray, y: float):
        self.t += 1
        self.X_hist.append(x)
        self.Y_hist.append(y)
        
        if len(self.X_hist) > self.window:
            self.X_hist = self.X_hist[-self.window:]
            self.Y_hist = self.Y_hist[-self.window:]
        
        if self.t >= self.lasso_start and self.t % self.lasso_period == 0:
            X_a = np.vstack(self.X_hist)
            y_a = np.array(self.Y_hist)
            alpha = self.lasso_alpha or np.sqrt(np.log(self.feature_dim) / max(1, X_a.shape[0]))
            lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
            lasso.fit(X_a, y_a)
            self.active_set = set(np.where(np.abs(lasso.coef_) > 1e-8)[0])

        idx = list(self.active_set)
        if idx:
            X_sub = np.vstack(self.X_hist)[:, idx]
            y_vec = np.array(self.Y_hist)
            coef = np.linalg.lstsq(X_sub, y_vec, rcond=None)[0]
            self.mu[:] = 0.0
            self.mu[idx] = coef

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        idx = list(self.active_set)
        if not idx:
            return 0.0, 0.0
            
        expected_reward = np.dot(x[idx], self.mu[idx])
        return expected_reward, 0.0

