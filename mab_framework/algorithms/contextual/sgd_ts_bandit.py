import numpy as np
from typing import List, Dict, Any
from ..base import BaseAlgorithm
from scipy.special import expit


class SGDTSBandit(BaseAlgorithm):
    """
    Online Thompson Sampling via Stochastic Gradient Descent (SGD-TS).
    
    References
    ----------
    Ding, W., Qi, Y., Lattimore, T., Zou, J., & Kpotufe, S. (2021). 
    "Provably efficient online Thompson sampling with linear payoffs 
    via stochastic gradient descent." arXiv preprint arXiv:2109.11762.
    """
    def __init__(self, d, K=2, model=None):
        super().__init__(K, model)
        self.original_d = d  # dimensionality of a single arm context
        self.d = K * d  # Total input dimension after expansion
        if self.model is None:
            raise ValueError("model must be provided to SGDTSBandit")

    def _expand_context(self, context):
        """
        Expand from shape (K, d) → (K, K*d) by placing each arm's context in its slot
        """
        expanded = np.zeros((self.n_arms, self.d))
        for k in range(self.n_arms):
            start = k * self.original_d
            end = (k + 1) * self.original_d
            expanded[k, start:end] = context[k]
        return expanded

    def select_arm(self, context: np.ndarray) -> int:
        expanded_context = self._expand_context(context)
        scores = self.model.sample(expanded_context)
        return int(np.argmax(scores))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            ctx = context[action] if context.ndim > 1 else context
            x = np.zeros(self.d)  # shape (K*d,)
            start = action * self.original_d
            end = (action + 1) * self.original_d
            x[start:end] = ctx

            self.model.fit(x, reward)

