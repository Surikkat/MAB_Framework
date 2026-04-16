import numpy as np
from typing import Union, List, Dict, Any
from .base import BaseAlgorithm
from BanditLab.models.base import BaseModel

class UCBAlgorithm(BaseAlgorithm):
    def __init__(self, n_arms: int, model: Union[BaseModel, List[BaseModel]], alpha: float = 1.0):
        super().__init__(n_arms, model)
        self.alpha = alpha

    def select_arm(self, context: np.ndarray) -> int:
        ucb_values = []
        for a in range(self.n_arms):
            x_a = context[a] if context.ndim > 1 else context
            mu, sigma = self.model[a].predict(x_a)
            ucb_values.append(mu + self.alpha * sigma)
        return int(np.argmax(ucb_values))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            x_a = context[action] if context.ndim > 1 else context
            self.model[action].fit(x_a, reward)

