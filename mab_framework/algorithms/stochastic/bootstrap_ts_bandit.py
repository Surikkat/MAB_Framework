import numpy as np
from typing import List, Dict, Any
from ..base import BaseAlgorithm
from ...models.bootstrap_model import BootstrapEnsembleModel


class BootstrapTSBandit(BaseAlgorithm):
    """
    Bootstrap Thompson Sampling Algorithm.
    
    Uses an ensemble of bootstrapped models to approximate the posterior 
    distribution for exploration.
    
    References
    ----------
    Osband, I., Blundell, C., Pritzel, A., & Van Roy, B. (2016). 
    "Deep Exploration via Bootstrapped DQN." NeurIPS.
    """
    def __init__(self, n_arms, d, model=None):
        super().__init__(n_arms, model)
        self.d = d
        if self.model is None or not isinstance(self.model, list):
            raise ValueError("model must be a list of BootstrapEnsembleModel (one per arm)")

    def select_arm(self, context):
        """
        context: shape (n_arms, d)
        """
        means = np.array([self.model[i].sample(context[i])[0] for i in range(self.n_arms)])
        return int(np.argmax(means))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            x_a = context[action] if context.ndim > 1 else context
            self.model[action].fit(x_a, reward)

