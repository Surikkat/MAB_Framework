import math
import numpy as np
from typing import List, Dict, Any
from .base import BaseAlgorithm
from models.base import BaseModel


class NNAGPUCBAlgorithm(BaseAlgorithm):
    """UCB algorithm for NN-AGP that uses a SINGLE shared model for all arms.

    Unlike UCBAlgorithm (which creates one model per arm), this algorithm
    matches the original NN-AGP paper where one GP posterior is shared
    across all arms. The model sees (theta, x) pairs from ALL arms and
    builds a unified posterior.

    NNAGPModel.predict() returns (mu, sigma²) — variance, not std.
    UCB = mu + sqrt(beta) * sqrt(sigma²), matching original:
        ucbs = mus + math.sqrt(self.beta) * np.sqrt(sig2s)

    Expects context of shape (n_arms, feature_dim) where each row is
    the concatenation [theta_t, x_a] for arm a.
    """

    def __init__(self, n_arms: int, model: BaseModel, beta: float = 2.0):
        super().__init__(n_arms, model)
        self.beta = beta

    def select_arm(self, context: np.ndarray) -> int:
        ucb_values = []
        for a in range(self.n_arms):
            x_a = context[a] if context.ndim > 1 else context
            mu, sigma2 = self.model.predict(x_a)
            ucb_values.append(mu + math.sqrt(self.beta) * np.sqrt(sigma2))
        return int(np.argmax(ucb_values))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            x_a = context[action] if context.ndim > 1 else context
            self.model.fit(x_a, reward, delay_fit=True)
            
        if len(feedbacks) > 0:
            self.model.finalize_update()

