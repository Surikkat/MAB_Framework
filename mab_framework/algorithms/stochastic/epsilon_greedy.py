import numpy as np
from typing import Union, List, Dict, Any
from ..base import BaseAlgorithm
from mab_framework.models.base import BaseModel

class EpsilonGreedy(BaseAlgorithm):
    def __init__(self, n_arms: int, model: Union[BaseModel, List[BaseModel]], epsilon: float = 0.1):
        super().__init__(n_arms, model)
        self.epsilon = epsilon

    def select_arm(self, context: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        values = []
        for a in range(self.n_arms):
            x_a = context[a] if context.ndim > 1 else context
            mu, _ = self.model[a].predict(x_a)
            values.append(mu)
        return int(np.argmax(values))

    def update(self, feedbacks: List[Dict[str, Any]]) -> None:
        for fb in feedbacks:
            action = fb["action"]
            reward = fb["reward"]
            context = fb["context"]
            x_a = context[action] if context.ndim > 1 else context
            self.model[action].fit(x_a, reward)