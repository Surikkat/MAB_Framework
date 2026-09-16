import numpy as np
from typing import Any, Optional
from mab_framework.models.base import BaseModel
from sklearn.linear_model import BayesianRidge
from scipy.stats import norm

class TabICLRegressorPPD(BaseModel):
    """
    Adapter for a Posterior Predictive Distribution (PPD) regressor 
    to implement the interface required by PFNTSAlgorithm.
    
    This implementation uses BayesianRidge from scikit-learn to provide
    a true mathematical PPD (mean and variance) instead of relying on
    the heavy/licensed tabpfn package.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._regressor = BayesianRidge()
        self.is_fitted = False
        self._mean = 0.0
        self._std = 1.0
        
    def fit(self, X, y, *args, **kwargs):
        if len(y) > 0:
            self._mean = np.mean(y)
            self._std = np.std(y) if len(y) > 1 else 1.0
            
            # BayesianRidge needs at least a few points to fit properly
            if len(y) >= 3:
                self._regressor.fit(X, y)
                self.is_fitted = True
            else:
                self.is_fitted = False
        return self
    
    def predict(self, X, *args, **kwargs):
        if not self.is_fitted:
            return np.full(len(X), self._mean), np.full(len(X), self._std)
        return self._regressor.predict(X, return_std=True)
    
    def condition(self, x_prev: np.ndarray, y_prev: np.ndarray) -> None:
        self.fit(x_prev, y_prev)
    
    def mean(self, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray) -> np.ndarray:
        self.condition(x_prev, y_prev)
        return self.mean_cached(x_new)
    
    def mean_cached(self, x_new: np.ndarray) -> np.ndarray:
        mean, _ = self.predict(x_new)
        return mean
        
    def cdf_cached(self, t: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """
        Computes the CDF evaluated at thresholds 't' for the given 'x_new'.
        Returns shape (len(t), len(x_new)).
        """
        mean, std = self.predict(x_new)
        
        # Ensure std is strictly positive to avoid division by zero
        std = np.maximum(std, 1e-6)
        
        # t is (len(t),)
        # mean is (len(x_new),)
        # We need output of shape (len(t), len(x_new))
        t_grid = t.reshape(-1, 1)
        mean_grid = mean.reshape(1, -1)
        std_grid = std.reshape(1, -1)
        
        # P(Y <= t) for Y ~ N(mean, std)
        return norm.cdf(t_grid, loc=mean_grid, scale=std_grid)
        
    def cdf(self, t: np.ndarray, x_new: np.ndarray, 
            x_prev: np.ndarray, y_prev: np.ndarray) -> np.ndarray:
        self.condition(x_prev, y_prev)
        return self.cdf_cached(t, x_new)
    
    def sample_cached(self, x_new: np.ndarray, 
                     rng: np.random.Generator = None, size: int = 1) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
            
        mean, std = self.predict(x_new)
        std = np.maximum(std, 1e-6)
        
        samples = np.zeros((size, len(x_new)))
        for i in range(len(x_new)):
            samples[:, i] = rng.normal(mean[i], std[i], size=size)
            
        return samples
        
    def sample(self, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray,
               rng: np.random.Generator = None, size: int = 1) -> np.ndarray:
        self.condition(x_prev, y_prev)
        return self.sample_cached(x_new, rng, size)
